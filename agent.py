"""
agent.py

Phase 7 — Native Tool Calling (ReAct) yetenekleri.

MİMARİ DEĞİŞİM (Phase 6 -> 7):
    ESKİ: graph.py'deki statik router soruyu ya SQL ya RAG yoluna gönderirdi
          ("either/or"). "Satışları VE iade politikasını ver" gibi bileşik
          sorular tek bir yola sıkışır, diğer kaynağa hiç bakılmazdı.
    YENİ: Yetenekler LangChain TOOL'larına dönüştürüldü. LLM, soruya göre
          bir VEYA BİRDEN FAZLA aracı (gerekirse aynı turda paralel olarak)
          çağırır. Bileşik sorular artık doğal biçimde desteklenir.

KORUNAN GÜVENLİK/DAYANIKLILIK GARANTİLERİ (hiçbiri kaybedilmedi):
  1. SELECT-only zırhı: _is_safe_select() — araç içinde, sorgu
     çalıştırılmadan ÖNCE. DROP/DELETE/UPDATE hâlâ imkânsız.
  2. Self-correction (Phase 4): Yeniden deneme döngüsü ARACIN İÇİNDE
     kaldı (MAX_RETRIES ile sınırlı, hata metni SQL üreticiye geri
     beslenir). Bilinçli tercih — gerekçesi query_database_tool içinde.
  3. Kaynak önceliği (Phase 6): tenant_db_url > tenant_db_path > varsayılan.
  4. Kimlik bilgisi izolasyonu: tenant_db_url araca InjectedState ile
     geçer; LLM'in gördüğü araç şemasında BULUNMAZ (test ile doğrulandı).
"""

import logging
from typing import Annotated, Optional

from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from typing_extensions import NotRequired, TypedDict

from database import execute_sql_query
from rag_node import retrieve_policy_node
from sql_generator import generate_sql_query

# Kurumsal Loglama Altyapısı
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ======================================================================
# 1. HAFIZA (STATE) TANIMI
# ======================================================================
class AgentState(TypedDict):
    """
    Tool-calling ajanının state şeması.

    messages: Konuşma geçmişi (add_messages reducer'ı ile birikir).
        Tool calling mimarisi mesaj tabanlıdır:
        HumanMessage -> AIMessage(tool_calls) -> ToolMessage(sonuç) -> ...
        Bu sayede ajan hem önceki turları hem de araç sonuçlarını görür.

    ⚠️ BÜYÜME KONTROLÜ: Araç çıktıları (SQL satırları, doküman parçaları)
    büyük olabilir ve her tur checkpoint'e yazılır — bu, Phase 5.2'de
    çözülen OOM riskinin geri dönmesi demek olurdu. graph.py'deki prune
    adımı, tur bitiminde ara araç trafiğini temizler; geçmişte yalnızca
    soru ve nihai cevap kalır.
    """

    messages: Annotated[list, add_messages]

    # --- Tenant-İzoleli Dinamik Veri (SaaS Modu) ---
    # NotRequired: varsayılan (demo veri) modda hiç bulunmayabilirler.
    # Araçlar bunları InjectedState üzerinden okur; LLM ASLA göremez/veremez.
    #
    # ⚠️ GÜVENLİK NOTU: tenant_db_url KİMLİK BİLGİSİ içerir. Bugün
    # checkpointer bellek içi (MemorySaver) olduğundan risk süreçle
    # sınırlıdır; kalıcı bir checkpointer'a geçilirse bağlantı dizeleri
    # DİSKE yazılır. O aşamada dizeyi state yerine süreç-içi bir kayıt
    # defterinde tutup state'te sadece anahtar taşımak gerekir.
    tenant_db_path: NotRequired[Optional[str]]
    tenant_db_url: NotRequired[Optional[str]]
    tenant_vector_store_path: NotRequired[Optional[str]]


# ======================================================================
# 2. SABİTLER VE ORTAK YARDIMCILAR
# ======================================================================

# Bir SQL sorusu için en fazla kaç kez SQL üretilip denenecek.
# (Phase 4'ten korunan self-correction bütçesi.)
MAX_RETRIES: int = 3

# Araç çıktılarındaki hata mesajlarının önekleri. is_error_result() bu tek
# listeye bakar; "bir sonuç hata mı?" sorusunun cevabı TEK yerde tanımlıdır.
_ERROR_PREFIXES: tuple[str, ...] = ("HATA:", "GÜVENLİK İHLALİ:")

# Araç çıktılarının başına eklenen kaynak etiketi. chat_engine.py bu
# etiketi ayrıştırarak "Kaynaklar" listesini DETERMİNİSTİK üretir —
# LLM'in kaynak hatırlamasına güvenilmez.
SOURCE_TAG_PREFIX: str = "[VERİ KAYNAĞI: "


def is_error_result(result: str) -> bool:
    """
    Bir metnin hata/başarısızlık mesajı olup olmadığını belirler.

    agent.py, graph.py ve chat_engine.py arasında paylaşılan TEK doğruluk
    kaynağıdır. Kaynak etiketi (SOURCE_TAG_PREFIX) başa eklenmiş olabilir;
    bu yüzden etiket atlanarak kontrol edilir.
    """
    if not result:
        return False
    text = result.strip()
    if text.startswith(SOURCE_TAG_PREFIX):
        # "[VERİ KAYNAĞI: X]\n..." -> ilk satırı atla
        parts = text.split("\n", 1)
        text = parts[1].strip() if len(parts) > 1 else ""
    return text.startswith(_ERROR_PREFIXES)


def _is_safe_select(query: str) -> bool:
    """
    🚨 GÜVENLİK ZIRHI (projenin en kritik değişmezi).

    Ajan yalnızca OKUMA yapabilir. Sorgu 'SELECT' ile başlamıyorsa
    çalıştırılmaz — DROP/DELETE/UPDATE/INSERT imkânsızdır. Bu kural
    kiracının KENDİ veritabanı ve canlı bağlantılar için de aynen geçerlidir.

    Not: Araçlar LLM'den ham SQL DEĞİL, DOĞAL DİL sorusu alır; SQL üretimi
    tamamen içeride yapılır. Yani saldırı yüzeyi zaten kapalıdır; bu
    kontrol ikinci savunma hattıdır (defense in depth).
    """
    return query.strip().upper().startswith("SELECT")


def _resolve_sql_source_label(
    tenant_db_url: Optional[str], tenant_db_path: Optional[str]
) -> str:
    """Aktif SQL veri kaynağının kullanıcıya gösterilecek adı."""
    if tenant_db_url:
        return "SQL (Canlı DB)"
    if tenant_db_path:
        return "SQL (Yüklenen DB)"
    return "SQL (Demo Veritabanı)"


def _tag(source_label: str, body: str) -> str:
    """Araç çıktısını kaynak etiketiyle sarar."""
    return f"{SOURCE_TAG_PREFIX}{source_label}]\n{body}"


# ======================================================================
# 3. ARAÇ 1: VERİTABANI SORGULAMA
# ======================================================================
@tool("query_database_tool")
def query_database_tool(question: str, state: Annotated[dict, InjectedState]) -> str:
    """Şirketin ilişkisel veritabanında (SQL) sorgulama yapar.

    Bu aracı; müşteriler, siparişler, ürünler, stok, fiyatlar, adetler,
    satış rakamları, tarihler gibi YAPILANDIRILMIŞ VERİ gerektiren
    sorular için kullan. Sayma, toplama, listeleme, filtreleme,
    sıralama ve karşılaştırma soruları buraya gider.

    Args:
        question: Cevaplanması istenen soruyu doğal dilde, tek başına
            anlaşılır biçimde yaz (örn. "en çok satan 5 ürün").
            SQL YAZMA — sorguyu sistem kendisi üretir.

    Returns:
        Veritabanından dönen satırlar veya açıklayıcı bir hata mesajı.
    """
    tenant_db_path = state.get("tenant_db_path") or None
    tenant_db_url = state.get("tenant_db_url") or None
    source_label = _resolve_sql_source_label(tenant_db_url, tenant_db_path)

    logging.info(f"🛠️ Araç: query_database_tool -> '{question[:80]}' ({source_label})")

    # --- SELF-CORRECTION DÖNGÜSÜ (Phase 4 mantığı, araç içine taşındı) ---
    #
    # Neden ReAct döngüsüne bırakılmadı: (a) ReAct'in yeniden denemesi
    # sınırsızdır, MAX_RETRIES gibi sert bir bütçesi yoktur; (b) hata
    # metnini SQL ÜRETİM prompt'una geri besleyemez — Phase 4'ün asıl
    # değeri buydu; (c) her deneme fazladan bir ajan LLM turu yakardı.
    # Araç içinde tutmak, test edilmiş davranışı birebir korur.
    previous_error: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            generated_sql = generate_sql_query(
                question,
                previous_error=previous_error,
                tenant_db_path=tenant_db_path,
                tenant_db_url=tenant_db_url,
            )
        except Exception as e:
            logging.error(f"SQL üretilirken kritik hata: {e}")
            return _tag(source_label, f"HATA: SQL üretilemedi. Detay: {e}")

        if not generated_sql:
            previous_error = "HATA: Boş sorgu üretildi."
            if attempt == MAX_RETRIES:
                return _tag(source_label, "HATA: Geçerli bir SQL sorgusu üretilemedi.")
            continue

        # 🚨 GÜVENLİK ZIRHI — çalıştırmadan ÖNCE
        if not _is_safe_select(generated_sql):
            logging.warning("🚨 Güvenlik zırhı devrede: SELECT olmayan sorgu engellendi.")
            return _tag(
                source_label,
                "GÜVENLİK İHLALİ: Ajan sadece SELECT (okuma) sorguları çalıştırabilir. "
                "DROP/DELETE/UPDATE yasaktır.",
            )

        try:
            rows = execute_sql_query(
                generated_sql,
                tenant_db_path=tenant_db_path,
                tenant_db_url=tenant_db_url,
            )
        except Exception as e:
            # Başarısızlık: hatayı bir sonraki ÜRETİM denemesine geri besle.
            previous_error = f"HATA: {e}"
            logging.warning(
                f"♻️ Self-correction (Deneme {attempt}/{MAX_RETRIES}): {str(e)[:120]}"
            )
            if attempt == MAX_RETRIES:
                return _tag(
                    source_label,
                    f"HATA: Sorgu {MAX_RETRIES} denemede çalıştırılamadı. Son hata: {e}",
                )
            continue

        # --- Başarı ---
        if not rows:
            return _tag(
                source_label,
                "Sorgu başarıyla çalıştı ancak hiçbir veri bulunamadı.\n"
                f"Çalıştırılan sorgu: {generated_sql}",
            )

        rows_text = "\n".join(str(row) for row in rows)
        return _tag(
            source_label,
            f"Çalıştırılan sorgu: {generated_sql}\n\nSonuçlar:\n{rows_text}",
        )

    # Döngü buraya normalde düşmez (her yol return eder); savunma amaçlı.
    return _tag(source_label, "HATA: Sorgu üretilemedi.")


# ======================================================================
# 4. ARAÇ 2: DOKÜMAN ARAMA (RAG)
# ======================================================================
@tool("search_documents_tool")
def search_documents_tool(question: str, state: Annotated[dict, InjectedState]) -> str:
    """Kurumsal dokümanlarda (PDF) anlamsal arama yapar.

    Bu aracı; iade politikası, kargo kuralları, garanti koşulları, ödeme
    seçenekleri, prosedürler, sözleşme maddeleri gibi YAZILI METİN ve
    POLİTİKA bilgisi gerektiren sorular için kullan.

    Args:
        question: Dokümanlarda aranacak konuyu doğal dilde, tek başına
            anlaşılır biçimde yaz (örn. "VIP müşteriler için iade süresi").

    Returns:
        En alakalı doküman bölümleri (dosya adı ve sayfa atıflarıyla)
        veya açıklayıcı bir hata mesajı.
    """
    tenant_store_path = state.get("tenant_vector_store_path") or None
    source_label = "Yüklenen Dokümanlar" if tenant_store_path else "Varsayılan Politikalar"

    logging.info(f"🛠️ Araç: search_documents_tool -> '{question[:80]}' ({source_label})")

    # rag_node'un düğüm sözleşmesi yeniden kullanılıyor: aynı retrieval
    # mantığı (kalıcı depoya bağlan, yeniden embedding YOK, çok dosyalı
    # atıflar, kayıp depoda net hata) hiç değişmeden korunur.
    node_output = retrieve_policy_node(
        {
            "question": question,
            "tenant_vector_store_path": tenant_store_path,
        }
    )
    return _tag(source_label, node_output.get("result", "") or "Sonuç bulunamadı.")


# ======================================================================
# 5. ARAÇ KAYDI
# ======================================================================
TOOLS = [query_database_tool, search_documents_tool]

__all__ = [
    "AgentState",
    "MAX_RETRIES",
    "SOURCE_TAG_PREFIX",
    "TOOLS",
    "is_error_result",
    "query_database_tool",
    "search_documents_tool",
]