"""
chat_engine.py

Phase 7 — Servis Katmanı / "Sözcü" (Middleware between graph and UI).

3 katmanlı mimarideki orta katman:

    app.py (Streamlit UI)  <->  chat_engine.py (Servis)  <->  graph.py (Motor)

PHASE 7 DEĞİŞİKLİKLERİ:
  1. Motor artık tool-calling grafiğidir. Bu katman, araç çağrısı ve araç
     sonucu olaylarını UI'ın bildiği aynı 4 tipe (status/meta/token/error)
     çevirir — app.py'nin sözleşmesi DEĞİŞMEDİ.
  2. Bileşik sorular: birden fazla araç çalıştığında her biri için ayrı
     durum mesajı yayınlanır ("Veritabanı sorgulanıyor...",
     "Dokümanlar taranıyor...").
  3. ATIF (CITATION) DÜZELTMESİ: Sözcü artık cümle aralarına dosya/sayfa
     referansı SERPİŞTİRMEZ. Metin akıcı yazılır; tüm kaynaklar cevabın
     EN SONUNDA düzenli bir "Kaynaklar:" listesinde toplanır. Bu liste
     LLM'den İSTENMEZ, araç çıktılarındaki gerçek etiketlerden
     DETERMİNİSTİK olarak üretilir (LLM'in kaynak hatırlamasına güvenilmez).
  4. Phase 6'nın deterministik sayısal grounding denetimi AYNEN korundu;
     denetim korpusu artık BU TURDA çalışan TÜM araçların çıktılarıdır.

YIELD SÖZLEŞMESİ (UI bu 4 tipten fazlasını bilmek zorunda değildir):
    {"type": "status", "message": str}   -> canlı ilerleme durumu
    {"type": "meta",   ...}              -> debug paneli için ham veriler
    {"type": "token",  "content": str}   -> sentezlenen cevabın parçası
    {"type": "error",  "message": str}   -> kurtarılamayan altyapı hatası
"""

import logging
import re
import threading
from typing import Generator, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent import SOURCE_TAG_PREFIX, AgentState, is_error_result
from database import redact_db_url
from graph import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Deterministik sayısal grounding kontrolü açık/kapalı.
# (Phase 6 güvenlik katmanı — kapatılırsa yalnızca olasılıksal prompt
# koruması kalır.)
GROUNDING_NUMERIC_CHECK: bool = True

# Cevabın sonuna eklenen kaynak listesinin başlığı.
SOURCES_HEADING: str = "Kaynaklar:"


# ======================================================================
# 1. GRAF YAŞAM DÖNGÜSÜ (Lazy Singleton)
# ======================================================================
# Derlenmiş graf (ve içindeki MemorySaver) süreç boyunca TEK olmalıdır:
# Streamlit her etkileşimde script'i yeniden çalıştırır; her seferinde
# yeni graf + yeni MemorySaver kurmak konuşma hafızasını sıfırlardı.
_graph_app = None
_graph_lock = threading.Lock()


def _get_graph_app():
    """Derlenmiş LangGraph uygulamasını döndürür (ilk çağrıda kurar)."""
    global _graph_app
    if _graph_app is not None:
        return _graph_app
    with _graph_lock:
        if _graph_app is None:
            _graph_app = build_graph()
    return _graph_app


# ======================================================================
# 2. ARAÇ -> DURUM MESAJI EŞLEMESİ
# ======================================================================
_TOOL_STATUS_MESSAGES: dict[str, str] = {
    "query_database_tool": "🗄️ Veritabanı sorgulanıyor...",
    "search_documents_tool": "📄 Dokümanlar taranıyor...",
}

_TOOL_STATUS_TENANT: dict[str, str] = {
    "query_database_tool": "🗄️ Bağlı veritabanınız sorgulanıyor...",
    "search_documents_tool": "📄 Yüklediğiniz dokümanlar taranıyor...",
}

_TOOL_DONE_MESSAGES: dict[str, str] = {
    "query_database_tool": "✅ Veritabanı sonuçları alındı.",
    "search_documents_tool": "✅ İlgili doküman bölümleri bulundu.",
}


# ======================================================================
# 3. SÖZCÜ (SPOKESPERSON) PROMPT'U — SERT GROUNDING + TEMİZ ATIF
# ======================================================================
# Halüsinasyon hatası (Phase 6): PDF'te "VIP iade süresi 45 gün" yazarken
# model kendi ön bilgisinden "30 gün" üretmişti. Aşağıdaki kurallar modeli
# iç ağırlıklarını değil YALNIZCA verilen bağlamı kullanmaya zorlar.
#
# Phase 7 eklentisi: metin içi atıf YASAĞI (okunabilirlik). Kaynak listesi
# ayrıca ve deterministik olarak kod tarafından eklenir.
_SPOKESPERSON_PROMPT = """Sen bir kurumsal veri asistanısın. Görevin, aşağıdaki HAM SONUÇ bölümünde verilen bilgiyi kibar, akıcı Türkçe ile aktarmaktır.

🔒 EN ÖNEMLİ KURAL — SIFIR TOLERANSLI KAYNAK BAĞLILIĞI:
Sen bir bilgi kaynağı DEĞİLSİN; sen yalnızca bir AKTARICISIN. Eğitim verinden gelen hiçbir bilgiyi kullanamazsın. Dünya genelinde "tipik", "yaygın", "standart" veya "genellikle böyledir" diye bildiğin hiçbir değer bu cevaba giremez. Tek gerçeklik kaynağın aşağıdaki HAM SONUÇ metnidir.

📏 SAYILAR VE METRİKLER İÇİN KATI KURALLAR:
1. Cevabında geçen HER sayı, tarih, süre, yüzde, para birimi ve ölçü, HAM SONUÇ içinde BİREBİR geçiyor olmalıdır. Bir sayıyı yazmadan önce onu HAM SONUÇ metninde gözünle bul.
2. ASLA yuvarlama, tahmin, ortalama alma veya birim çevirme yapma. HAM SONUÇ "45 gün" diyorsa cevabın "45 gün" olmalıdır — "yaklaşık 1,5 ay" veya "30 gün" değil.
3. HAM SONUÇ'ta olmayan bir sayıyı, "mantıklı görünüyor" diye ASLA ekleme.
4. Sorulan bilgi HAM SONUÇ'ta hiç yoksa, tam olarak şunu yaz: "Veri bulunamadı." Ardından hangi bilginin eksik olduğunu tek cümleyle belirt. Boşluğu doldurmaya ÇALIŞMA.
5. HAM SONUÇ birbiriyle çelişen iki değer içeriyorsa (örneğin farklı dokümanlarda farklı süreler), ikisini de belirt ve aralarında SEÇİM YAPMA.
6. HAM SONUÇ'ta özel bir koşul belirtiliyorsa (örneğin "VIP müşteriler için", "500 TL üzeri siparişlerde"), bu koşulu cevabında AYNEN koru. Koşulu düşürüp değeri genelleştirmek, yanlış bilgi vermektir.

📎 ATIF KURALLARI (OKUNABİLİRLİK — BUNA MUTLAKA UY):
7. Cevap metninin İÇİNE ASLA kaynak referansı YAZMA. Cümlelerin arasında veya sonunda "(rapor.pdf, Sayfa 3)", "[Doküman 1]", "kaynak: ...", "Sayfa 5'e göre" gibi ifadeler KULLANMA. Bu ifadeler metnin akıcılığını bozar.
8. Kaynak listesini de SEN YAZMA. Cevabının sonuna "Kaynaklar" başlığı veya kaynak listesi EKLEME — bu liste sistem tarafından otomatik olarak eklenecektir.
9. Sadece doğal, akıcı, kaynak referansı içermeyen düz bir Türkçe metin yaz.
10. Birden fazla veri kaynağından bilgi geldiyse (örneğin hem veritabanı hem doküman), bunları tek ve bütünlüklü bir cevapta birleştir; ayrı ayrı bölümler halinde parçalama, ama hiçbir bilgiyi de atlama.

⚠️ HATA VE BOŞ SONUÇ DURUMLARI:
- HAM SONUÇ "HATA:" veya "GÜVENLİK İHLALİ:" ile başlıyorsa: kullanıcıdan kibarca özür dile, isteğini şu anda yanıtlayamadığını sade bir dille açıkla. Teknik ayrıntıları (stack trace, SQL hata metni, dosya yolu, bağlantı bilgisi) ASLA aktarma. Hata mesajı kullanıcının yapabileceği bir eylem öneriyorsa (örneğin "dosyaları yeniden yükleyin"), bu öneriyi kibarca ilet.
- HAM SONUÇ "veri bulunamadı" diyorsa: kayda ulaşılamadığını nazikçe belirt ve olmayan veriyi UYDURMA.
- HAM SONUÇ veritabanı satırları içeriyorsa (parantezli tuple'lar vb.): bunları okunabilir bir liste/metin hâline getir; değerleri birebir koru, yeni değer ekleme.

✍️ ÜSLUP:
Kısa ve öz ol, doğrudan cevaba geç, selamlaşma tekrarı yapma. Kibar ve profesyonel bir dil kullan.

KULLANICININ SORUSU:
{question}

HAM SONUÇ:
{raw_result}

KİBAR, KAYNAĞA BİREBİR BAĞLI, METİN İÇİ ATIF İÇERMEYEN TÜRKÇE CEVABIN:"""


def _stream_spokesperson(question: str, raw_result: str) -> Generator[str, None, None]:
    """
    Sözcü LLM'i streaming modda çalıştırır ve cevap token'larını üretir.
    (Testlerde monkeypatch'lenebilsin diye ayrı fonksiyon.)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    prompt = ChatPromptTemplate.from_template(_SPOKESPERSON_PROMPT)
    chain = prompt | llm

    for chunk in chain.stream({"question": question, "raw_result": raw_result}):
        content = getattr(chunk, "content", "")
        if content:
            yield content


# ======================================================================
# 4. DETERMİNİSTİK GROUNDING DOĞRULAMASI (Phase 6 — korunuyor)
# ======================================================================
# Prompt sertleştirmesi halüsinasyon OLASILIĞINI düşürür ama sıfırlamayı
# GARANTİ EDEMEZ (LLM davranışı olasılıksaldır). Bu katman, prompt'a
# güvenmeyen mekanik bir kontroldür: cevaptaki her sayı, bağlamda veya
# soruda geçiyor mu?
_NUMBER_PATTERN = re.compile(r"\d[\d.,]*")
_LIST_MARKER_PATTERN = re.compile(r"(?m)^\s*\d+[.)]\s")


def _normalize_number(token: str) -> str:
    """'1.500' ve '1500' aynı sayıdır; ayırıcıları ve sondaki noktalamayı at."""
    return token.replace(".", "").replace(",", "").rstrip("0123456789.,") or token.replace(
        ".", ""
    ).replace(",", "")


def _extract_numbers(text: str) -> set[str]:
    """Metindeki tüm sayıları normalize edilmiş biçimde döndürür."""
    return {
        _normalize_number(m.group(0))
        for m in _NUMBER_PATTERN.finditer(text)
        if _normalize_number(m.group(0))
    }


def _find_ungrounded_numbers(answer: str, raw_result: str, question: str) -> list[str]:
    """
    Cevapta geçip HAM SONUÇ'ta (veya kullanıcının sorusunda) geçmeyen
    sayıları döndürür.

    Liste madde numaraları ("1.", "2)") sayılmaz — bunlar biçimlendirme
    artefaktıdır, veri iddiası değildir.

    Bu kontrol kasıtlı olarak SAYILARLA sınırlıdır: hatalı iddiaların en
    zararlı ve en kolay tespit edilebilir biçimi sayısal olanlardır
    (süreler, ücretler, oranlar). Serbest metin iddialarının doğrulanması
    ayrı bir problem alanıdır (NLI temelli kontrol — sonraki faz adayı).
    """
    answer_wo_markers = _LIST_MARKER_PATTERN.sub("", answer)
    grounded = _extract_numbers(raw_result) | _extract_numbers(question)
    return sorted(n for n in _extract_numbers(answer_wo_markers) if n not in grounded)


# ======================================================================
# 5. ATIF (CITATION) TOPLAMA — DETERMİNİSTİK
# ======================================================================
# Araç çıktılarındaki gerçek etiketlerden kaynak listesi üretilir.
# LLM'e "kaynakları listele" demek olasılıksaldır (unutabilir, uydurabilir);
# bu yüzden liste KOD tarafından üretilir.
_OUTER_SOURCE_PATTERN = re.compile(r"\[VERİ KAYNAĞI:\s*([^\]]+)\]")
_INNER_DOC_PATTERN = re.compile(r"\[Doküman\s+\d+\s*\|\s*Kaynak:\s*([^\]]+)\]")


def _format_document_source(raw_label: str) -> str:
    """
    Doküman etiketini okunur kaynak satırına çevirir.

    "vip.pdf, Sayfa 3"   -> "vip.pdf (Sayfa 3)"
    "iade_politikasi"    -> "iade_politikasi"
    """
    label = raw_label.strip()
    match = re.match(r"^(.*?),\s*(Sayfa\s+\d+)$", label)
    if match:
        return f"{match.group(1).strip()} ({match.group(2).strip()})"
    return label


def _collect_sources(tool_outputs: list[str]) -> list[str]:
    """
    Araç çıktılarından, sırayı koruyarak tekilleştirilmiş kaynak listesi üretir.

    - Doküman aracı çıktısında iç etiketler varsa (dosya + sayfa), onlar
      kullanılır; böylece "vip.pdf (Sayfa 3)" gibi hassas atıf elde edilir.
    - Aksi hâlde dış etiket (örn. "SQL (Canlı DB)") kullanılır.
    - Hata döndüren araç çıktıları kaynak listesine ALINMAZ; başarısız bir
      çağrıyı kaynak diye göstermek yanıltıcı olurdu.
    """
    sources: list[str] = []

    for output in tool_outputs:
        if not output or is_error_result(output):
            continue

        inner_labels = _INNER_DOC_PATTERN.findall(output)
        if inner_labels:
            for raw in inner_labels:
                formatted = _format_document_source(raw)
                if formatted and formatted not in sources:
                    sources.append(formatted)
            continue

        outer = _OUTER_SOURCE_PATTERN.search(output)
        if outer:
            label = outer.group(1).strip()
            if label and label not in sources:
                sources.append(label)

    return sources


def _build_sources_block(sources: list[str]) -> str:
    """Kaynak listesini cevabın sonuna eklenecek metin bloğuna çevirir."""
    if not sources:
        return ""
    lines = "\n".join(f"- {s}" for s in sources)
    return f"\n\n**{SOURCES_HEADING}**\n{lines}"


def _strip_inline_citations(answer: str, sources: list[str]) -> str:
    """
    Emniyet ağı: Sözcü kurala rağmen metin içine atıf serpiştirdiyse,
    YALNIZCA gerçekten bildiğimiz kaynaklara işaret eden parantezli
    referansları temizler.

    Kasıtlı olarak dar kapsamlı: rastgele parantez içeriğine dokunmaz
    (kullanıcının verisinde parantez olabilir), sadece bilinen dosya
    adlarını ve "[Doküman N]" kalıbını hedefler.
    """
    cleaned = re.sub(r"\[\s*Doküman\s+\d+[^\]]*\]", "", answer)

    file_names = {
        re.match(r"^(.*?)\s*\(Sayfa", s).group(1).strip() if "(Sayfa" in s else s
        for s in sources
    }
    for name in file_names:
        if not name or len(name) < 4:
            continue
        escaped = re.escape(name)
        # "(rapor.pdf, Sayfa 3)" / "(rapor.pdf)" biçimlerini kaldır
        cleaned = re.sub(rf"\s*\(\s*{escaped}[^)]*\)", "", cleaned)

    # Temizlik sonrası oluşabilecek çift boşlukları sadeleştir
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


# ======================================================================
# 6. ANA SERVİS FONKSİYONU
# ======================================================================
def stream_chat_response(
    question: str,
    thread_id: str,
    tenant_db_path: Optional[str] = None,
    tenant_db_url: Optional[str] = None,
    tenant_vector_store_path: Optional[str] = None,
) -> Generator[dict, None, None]:
    """
    Bir kullanıcı sorusunu uçtan uca işler ve UI'ın tüketeceği olay
    akışını üretir: durum mesajları -> meta (debug) -> cevap token'ları.

    Args:
        question: Kullanıcının doğal dildeki sorusu.
        thread_id: Konuşma oturumu kimliği (MemorySaver hafıza anahtarı).
        tenant_db_path: Yüklenen SQLite dosyasının yolu.
        tenant_db_url: Canlı veritabanı bağlantı dizesi (en yüksek öncelik).
        tenant_vector_store_path: Yükleme anında vektörlenmiş kalıcı
            Chroma deposunun dizin yolu.

    Yields:
        Modül docstring'indeki YIELD SÖZLEŞMESİ'ne uyan sözlükler.
    """
    question = (question or "").strip()
    if not question:
        yield {"type": "error", "message": "Lütfen bir soru yazın."}
        return

    config = {"configurable": {"thread_id": thread_id}}
    tenant_sql_active = bool(tenant_db_url or tenant_db_path)

    # Tenant anahtarları HER turda açıkça yazılır. Gerekçe: checkpointer,
    # girdide bulunmayan anahtarların ESKİ değerini korur (deneysel olarak
    # doğrulandı) — kullanıcı bir kaynağı kaldırdığında None yazmazsak
    # eski referans "hayalet veri" olarak yaşamaya devam ederdi.
    #
    # messages ise BİLEREK sıfırlanmaz: add_messages reducer'ı yeni soruyu
    # mevcut geçmişin ÜSTÜNE ekler; konuşma sürekliliği böyle sağlanır.
    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "tenant_db_path": tenant_db_path,
        "tenant_db_url": tenant_db_url,
        "tenant_vector_store_path": tenant_vector_store_path,
    }

    graph_app = _get_graph_app()

    tool_outputs: list[str] = []
    tools_used: list[str] = []

    # ------------------------------------------------------------------
    # ADIM A: Grafı akış modunda çalıştır, araç olaylarını duruma çevir
    # ------------------------------------------------------------------
    yield {"type": "status", "message": "🧭 Sorunuz analiz ediliyor..."}

    try:
        for event in graph_app.stream(initial_state, config, stream_mode="updates"):
            for node_name, update in event.items():
                messages = (update or {}).get("messages", []) or []

                # --- Ajan düğümü: hangi araçları çağırmaya karar verdi? ---
                if node_name == "agent":
                    for message in messages:
                        for call in getattr(message, "tool_calls", None) or []:
                            tool_name = call.get("name", "")
                            status_map = (
                                _TOOL_STATUS_TENANT
                                if (tenant_sql_active or tenant_vector_store_path)
                                else _TOOL_STATUS_MESSAGES
                            )
                            status = status_map.get(
                                tool_name, f"🛠️ '{tool_name}' çalıştırılıyor..."
                            )
                            yield {"type": "status", "message": status}

                # --- Araç düğümü: sonuçları topla ---
                elif node_name == "tools":
                    for message in messages:
                        if not isinstance(message, ToolMessage):
                            continue
                        content = message.content if isinstance(message.content, str) else str(message.content)
                        tool_outputs.append(content)
                        if message.name:
                            tools_used.append(message.name)
                            done = _TOOL_DONE_MESSAGES.get(message.name)
                            if done:
                                yield {"type": "status", "message": done}

    except Exception as e:
        logging.error(f"Graf çalıştırılırken kritik hata: {e}")
        yield {
            "type": "error",
            "message": (
                "Üzgünüm, isteğiniz işlenirken beklenmedik bir sistem hatası oluştu. "
                "Lütfen birazdan tekrar deneyin."
            ),
        }
        return

    # ------------------------------------------------------------------
    # ADIM B: Grounding korpusu ve kaynaklar
    # ------------------------------------------------------------------
    # ÖNEMLİ: Araç çıktıları AKIŞ SIRASINDA toplanır, nihai state'ten
    # okunmaz — graph.py'deki prune düğümü ara araç mesajlarını geçmişten
    # bilinçli olarak siler (checkpoint şişmesini önlemek için).
    if tool_outputs:
        raw_result = "\n\n".join(tool_outputs)
    else:
        # Hiç araç çağrılmadı: selamlaşma/sohbet ya da kapsam dışı soru.
        raw_result = (
            "Bu soru için herhangi bir veri kaynağına başvurulmadı. "
            "Kullanıcıya, yalnızca yüklü veritabanı ve dokümanlar hakkında "
            "soru yanıtlayabildiğini kibarca açıkla."
        )

    sources = _collect_sources(tool_outputs)
    any_error = any(is_error_result(o) for o in tool_outputs)

    yield {
        "type": "meta",
        "source": ", ".join(sources) if sources else "Sistem",
        "tools_used": tools_used,
        "raw_result": raw_result,
        "sources": sources,
        "is_error": any_error,
        "tenant_db": bool(tenant_db_path),
        "tenant_db_url": redact_db_url(tenant_db_url) if tenant_db_url else None,
        "tenant_doc": bool(tenant_vector_store_path),
    }

    # ------------------------------------------------------------------
    # ADIM C: Sözcü ile nihai cevabı token token akıt
    # ------------------------------------------------------------------
    yield {"type": "status", "message": "✍️ Cevabınız hazırlanıyor..."}

    answer_parts: list[str] = []

    try:
        for token in _stream_spokesperson(question, raw_result):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
    except Exception as e:
        logging.error(f"Sözcü LLM akışı başarısız: {e}")
        # Statik yedek cevap — kullanıcı asla cevapsız kalmaz.
        if any_error or not tool_outputs:
            fallback = (
                "Üzgünüm, isteğinizi şu anda yanıtlayamıyorum. "
                "Lütfen sorunuzu farklı bir şekilde ifade ederek tekrar dener misiniz?"
            )
        else:
            fallback = f"Sistemden gelen sonuç:\n\n{raw_result}"
        answer_parts.append(fallback)
        yield {"type": "token", "content": fallback}

    answer_text = "".join(answer_parts)

    # --- Emniyet ağı: metin içi atıf sızdıysa temizle ---
    cleaned = _strip_inline_citations(answer_text, sources)
    if cleaned != answer_text.strip():
        logging.info("🧹 Metin içi atıf(lar) temizlendi (okunabilirlik kuralı).")
        # Akış zaten ekrana yazıldığı için farkı düzeltmek yerine, UI'ın
        # nihai metni kullanabilmesi adına 'replace' sinyali gönderiyoruz.
        yield {"type": "replace", "content": cleaned}
        answer_text = cleaned

    # --- DETERMİNİSTİK GROUNDING DOĞRULAMASI (Phase 6 — korunuyor) ---
    if GROUNDING_NUMERIC_CHECK and not any_error:
        ungrounded = _find_ungrounded_numbers(answer_text, raw_result, question)
        if ungrounded:
            logging.warning(
                f"⚠️ GROUNDING UYARISI: cevapta kaynakta bulunmayan sayı(lar): {ungrounded}"
            )
            yield {
                "type": "token",
                "content": (
                    "\n\n> ⚠️ **Doğrulama uyarısı:** Bu cevaptaki şu değer(ler) kaynak "
                    f"verilerde bulunamadı: {', '.join(ungrounded)}. Lütfen bu bilgiyi "
                    "orijinal kaynaktan teyit edin."
                ),
            }

    # --- KAYNAKLAR BLOĞU (deterministik, cevabın EN SONUNDA) ---
    sources_block = _build_sources_block(sources)
    if sources_block:
        yield {"type": "token", "content": sources_block}