"""
rag_node.py

Phase 2 + 5.2 + 6 — Kurumsal Doküman Zekâsı (RAG) + Çok Dosyalı Kalıcı Depo.

Bu modül hem LangGraph retrieval düğümünü hem de TENANT DOKÜMAN DEPOSU
yaşam döngüsünü (ingestion, bağlanma, çöp toplama) barındırır. UI katmanı
(app.py) bu modüldeki fonksiyonları ÇAĞIRIR ama işin nasıl yapıldığını
bilmez (Separation of Concerns).

MİMARİ (5.2): Upload-Time Vectorization
  PDF'ler YÜKLEME ANINDA BİR KEZ işlenir: parçala -> embed et -> diske yaz.
  State yalnızca deponun küçük yolunu taşır; soru anında depoya SADECE
  bağlanılır (doküman yeniden embed EDİLMEZ, sadece sorgu vektörlenir).

PHASE 6 EKLENTİSİ: Çok dosyalı yükleme
  - ingest_pdfs_to_persistent_store(): birden çok PDF'i TEK ve YENİ bir
    depoya, TEK yazma işlemiyle dizinler. Aynı dizin tekrar tekrar açılıp
    kapanmadığı için dosya kilidi oluşmaz; ayrıca kısmi/bozuk depo riski
    yoktur (ya hepsi yazılır ya hiçbiri).
  - Her parçaya kaynak dosya adı (source_file) metadata olarak eklenir;
    atıflar "rapor.pdf, Sayfa 3" biçiminde döner. Bu, hem kullanıcı
    güveni hem de Sözcü'nün doğru dosyaya dayanması için önemlidir.
  - manifest.json: depodaki dosyaların listesi. UI bunu gösterir,
    retrieval ise TOP-K'yı buna göre uyarlar (aşağıya bakınız).

ZOMBIE DOSYA SAVUNMASI (çok katmanlı — sadece atexit'e güvenilmez):
  1. Değiştirme/kaldırma anında anında silme (app.py çağırır),
  2. Her uygulama açılışında cleanup_stale_tenant_data(): 12 saatten
     eski oturum dizinleri shutil.rmtree ile süpürülür — sunucu sert
     çökse bile bir sonraki açılış temizler,
  3. Aktif oturumlar her etkileşimde dizinlerine "dokunur" (mtime
     yenileme) — GC, yaşayan bir oturumun deposunu asla süpürmez.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from typing import Optional, Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Kurumsal Loglama Altyapısı (diğer modüllerle aynı format)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ======================================================================
# 1. KONFİGÜRASYON SABİTLERİ
# ======================================================================

# Tek dokümanlı aramada döndürülecek en alakalı parça sayısı.
TOP_K_RESULTS: int = 3

# Çok dosyalı depolarda üst sınır. Gerekçe: 5 PDF'lik bir depoda k=3,
# doğru cevabı içeren parçanın hiç getirilmemesine (ve Sözcü'nün
# "Veri bulunamadı" demesine ya da daha kötüsü uydurmasına) yol açabilir.
MAX_TOP_K_RESULTS: int = 8

# Varsayılan (dummy politika) koleksiyonunun adı.
COLLECTION_NAME: str = "kurumsal_politikalar"

# Tenant dokümanlarının kalıcı Chroma koleksiyon adı.
TENANT_COLLECTION_NAME: str = "tenant_dokuman"

# Depodaki dosyaların listesini tutan manifest dosyası.
MANIFEST_FILENAME: str = "manifest.json"

# Embedding modeli. text-embedding-3-small: düşük maliyet, bu ölçek için
# fazlasıyla yeterli doğruluk.
EMBEDDING_MODEL: str = "text-embedding-3-small"

# Tenant PDF'leri için metin parçalama ayarları.
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

# Tenant oturum verilerinin (vektör depoları + geçici DB dosyaları) kökü.
TENANT_DATA_ROOT: str = os.path.join(".", "temp_tenant_data")

# Bu yaştan eski oturum dizinleri açılışta süpürülür.
STALE_SESSION_MAX_AGE_HOURS: float = 12.0


# ======================================================================
# 2. DUMMY KURUMSAL VERİ (Varsayılan mod — Phase 2'den değişmedi)
# ======================================================================
_DUMMY_POLICY_DOCUMENTS: list[Document] = [
    Document(
        page_content=(
            "İade politikası: Ürünler teslim alındıktan sonra 14 gün içinde "
            "orijinal ambalajında iade edilebilir. İade kargo ücreti, ürün "
            "kusurluysa şirketimiz tarafından karşılanır."
        ),
        metadata={"kaynak": "iade_politikasi", "kategori": "musteri_hizmetleri"},
    ),
    Document(
        page_content=(
            "Kargo kuralları: 500 TL üzeri alışverişlerde kargo ücretsizdir, "
            "aksi takdirde standart kargo ücreti 50 TL'dir. Siparişler 1-3 iş "
            "günü içinde kargoya verilir."
        ),
        metadata={"kaynak": "kargo_kurallari", "kategori": "lojistik"},
    ),
    Document(
        page_content=(
            "Garanti koşulları: Tüm elektronik ürünler 2 yıl resmi distribütör "
            "garantisi kapsamındadır. Garanti, kullanıcı hatasından kaynaklanan "
            "hasarları kapsamaz."
        ),
        metadata={"kaynak": "garanti_kosullari", "kategori": "musteri_hizmetleri"},
    ),
    Document(
        page_content=(
            "Ödeme seçenekleri: Kredi kartına 12 aya varan taksit imkânı "
            "sunulmaktadır. Kapıda ödeme hizmet bedeli 25 TL'dir. Havale/EFT "
            "ödemelerinde %3 indirim uygulanır."
        ),
        metadata={"kaynak": "odeme_secenekleri", "kategori": "finans"},
    ),
]


# ======================================================================
# 3. EMBEDDING SAĞLAYICI
# ======================================================================
def _get_embeddings() -> Embeddings:
    """
    Embedding sağlayıcısını döndürür.

    Ayrı fonksiyon olmasının nedeni test edilebilirliktir: birim testler
    rag_node._get_embeddings'i monkeypatch'leyerek gerçek API'ye gitmeden
    tüm akışı doğrulayabilir.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY bulunamadı! RAG modülü embedding üretmek için "
            "OpenAI API'ye ihtiyaç duyar. Lütfen .env dosyanızı kontrol edin."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ======================================================================
# 4. VARSAYILAN VECTOR STORE YAŞAM DÖNGÜSÜ (Lazy Singleton — değişmedi)
# ======================================================================
_vector_store: Optional[Chroma] = None
_init_lock = threading.Lock()


def get_vector_store(embeddings: Optional[Embeddings] = None) -> Chroma:
    """
    Varsayılan (dummy politika) in-memory vector store'u döndürür;
    ilk çağrıda kurar, sonraki çağrılarda mevcut tekil örneği verir.
    """
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    with _init_lock:
        if _vector_store is not None:
            return _vector_store

        if embeddings is None:
            embeddings = _get_embeddings()

        logging.info(
            f"📚 RAG: Varsayılan in-memory ChromaDB kuruluyor "
            f"({len(_DUMMY_POLICY_DOCUMENTS)} politika dokümanı embed edilecek)..."
        )

        _vector_store = Chroma.from_documents(
            documents=_DUMMY_POLICY_DOCUMENTS,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
        )

        logging.info("✅ RAG: Varsayılan vector store hazır.")
        return _vector_store


def reset_vector_store() -> None:
    """Tekil store referansını sıfırlar (öncelikle testler için)."""
    global _vector_store
    with _init_lock:
        _vector_store = None


# ======================================================================
# 5. TENANT DOKÜMAN DEPOSU YAŞAM DÖNGÜSÜ
# ======================================================================
def get_tenant_session_dir(session_data_id: str) -> str:
    """
    Bir oturumun tüm tenant verilerinin (vektör deposu + DB dosyaları)
    yaşayacağı dizin yolunu döndürür ve dizinin var olmasını garanti eder.
    """
    session_dir = os.path.join(TENANT_DATA_ROOT, session_data_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def touch_tenant_dir(session_dir: str) -> None:
    """
    Oturum dizininin mtime'ını şimdiye çeker ("dokunma").

    Aktif oturumlar her etkileşimde bunu çağırır; böylece 12 saatlik GC
    eşiği "son kullanımdan beri" işler ve uzun yaşayan aktif bir oturumun
    deposu asla süpürülmez.
    """
    try:
        if os.path.isdir(session_dir):
            os.utime(session_dir, None)
    except OSError:
        pass  # Dokunamamak kritik değildir.


def cleanup_stale_tenant_data(
    base_dir: str = TENANT_DATA_ROOT,
    max_age_hours: float = STALE_SESSION_MAX_AGE_HOURS,
) -> int:
    """
    Açılış Çöp Toplayıcısı (Startup Garbage Collector).

    base_dir altındaki, mtime'ı max_age_hours'tan eski TÜM oturum
    dizinlerini shutil.rmtree ile siler. Sunucu daha önce sert çökmüş
    (atexit çalışamamış) olsa bile, bir sonraki açılışta ortalık temizlenir.

    Güvenlik notları:
      - SADECE base_dir'in DOĞRUDAN alt dizinleri değerlendirilir.
      - Silme hataları yutulur ve loglanır — GC açılışı asla engellememelidir.

    Returns:
        Silinen oturum dizini sayısı.
    """
    if not os.path.isdir(base_dir):
        return 0

    cutoff = time.time() - (max_age_hours * 3600)
    deleted = 0

    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            if os.path.getmtime(entry_path) < cutoff:
                shutil.rmtree(entry_path, ignore_errors=True)
                deleted += 1
                logging.info(f"🧹 GC: Eski tenant oturum dizini silindi -> {entry_path}")
        except OSError as e:
            logging.warning(f"⚠️ GC: '{entry_path}' değerlendirilemedi: {e}")

    if deleted:
        logging.info(f"🧹 GC: Toplam {deleted} eski oturum dizini temizlendi.")
    return deleted


def remove_tenant_vector_store(store_path: Optional[str]) -> None:
    """
    Bir tenant vektör deposunu diskten anında siler (değiştirme/kaldırma
    akışları için). Yol yoksa sessizce geçer.
    """
    try:
        if store_path and os.path.isdir(store_path):
            shutil.rmtree(store_path, ignore_errors=True)
            logging.info("🧹 Tenant vektör deposu diskten silindi.")
    except OSError as e:
        logging.warning(f"⚠️ Vektör deposu silinemedi (GC daha sonra süpürecek): {e}")


def _write_manifest(persist_directory: str, file_names: list[str], chunk_count: int) -> None:
    """Depodaki kaynak dosyaların listesini manifest.json olarak yazar."""
    manifest = {"files": file_names, "chunks": chunk_count, "created_at": time.time()}
    try:
        with open(os.path.join(persist_directory, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)
    except OSError as e:
        # Manifest kritik değildir (UI/k-ayarı için yardımcıdır); yokluğu
        # aramayı bozmaz, yalnızca varsayılan davranışa düşülür.
        logging.warning(f"⚠️ Manifest yazılamadı: {e}")


def get_tenant_manifest(persist_directory: Optional[str]) -> dict:
    """
    Depodaki dosya listesini döndürür ({"files": [...], "chunks": n}).
    Manifest yoksa boş yapı döner — çağıranlar buna dayanabilir.
    """
    if not persist_directory:
        return {"files": [], "chunks": 0}
    try:
        with open(os.path.join(persist_directory, MANIFEST_FILENAME), encoding="utf-8") as f:
            data = json.load(f)
            return {"files": list(data.get("files", [])), "chunks": int(data.get("chunks", 0))}
    except (OSError, ValueError):
        return {"files": [], "chunks": 0}


def _pdf_bytes_to_documents(pdf_bytes: bytes, source_name: str = "Yüklenen PDF") -> list[Document]:
    """
    Ham PDF byte'larını LangChain Document (chunk) listesine çevirir ve
    her parçaya kaynak dosya adını metadata olarak işler.

    tempfile yaşam döngüsü:
      - NamedTemporaryFile(delete=False) + finally os.unlink kalıbı
        bilinçli tercih: `delete=True` iken dosyayı adıyla ikinci kez
        açmak (PyPDFLoader'ın yaptığı) Windows'ta kilit hatası verir.
        delete=False + finally, tüm platformlarda hem okumayı hem de
        GARANTİLİ silinmeyi sağlar — yükleme patlasa bile dosya silinir.

    Raises:
        ValueError: PDF'ten hiç metin çıkarılamadıysa (örn. taranmış
            görüntü-PDF) — çağırana net, eyleme dönük bir hata verilir.
    """
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logging.info("🧹 Geçici PDF dosyası güvenle silindi.")

    if not pages or not any(p.page_content.strip() for p in pages):
        raise ValueError(
            f"'{source_name}' dosyasından okunabilir metin çıkarılamadı. "
            "Dosya taranmış bir görüntü olabilir; lütfen metin tabanlı bir PDF yükleyin."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    # Kaynak dosya adını her parçaya işle -> çok dosyalı depoda atıflar
    # "hangi dosyanın hangi sayfası" olarak dönebilir.
    for chunk in chunks:
        chunk.metadata["source_file"] = source_name

    logging.info(f"📄 '{source_name}': {len(pages)} sayfa -> {len(chunks)} metin parçası.")
    return chunks


def ingest_pdfs_to_persistent_store(
    files: Sequence[tuple[str, bytes]],
    persist_directory: str,
    embeddings: Optional[Embeddings] = None,
) -> dict:
    """
    ÇOK DOSYALI UPLOAD-TIME VECTORIZATION.

    Verilen TÜM PDF'leri tek bir kalıcı Chroma deposuna, TEK yazma
    işlemiyle dizinler. Tasarım gerekçeleri:
      - Dosya kilidi yok: hedef dizin yalnızca bir kez açılır; her dosya
        için depoyu tekrar tekrar açıp kapatmayız.
      - Kısmi depo yok: parçalar önce bellekte toplanır, sonra tek seferde
        yazılır. Yazma patlarsa dizin komple silinir.
      - Dayanıklılık: bir dosya okunamazsa (taranmış PDF vb.) diğerleri
        yine dizinlenir; başarısızlar rapor edilir. Kullanıcı 5 dosyadan
        1'i bozuk diye hepsini yeniden yüklemek zorunda kalmaz.

    Args:
        files: (dosya_adı, pdf_byte) ikilileri.
        persist_directory: Deponun yazılacağı YENİ dizin.
        embeddings: Test enjeksiyonu için opsiyonel override.

    Returns:
        {"chunk_count": int, "ingested": [ad, ...], "failed": [(ad, sebep), ...]}

    Raises:
        ValueError: Hiçbir dosyadan tek bir parça bile çıkarılamadıysa.
    """
    if not files:
        raise ValueError("Dizinlenecek dosya bulunamadı.")

    all_chunks: list[Document] = []
    ingested: list[str] = []
    failed: list[tuple[str, str]] = []

    for file_name, pdf_bytes in files:
        try:
            chunks = _pdf_bytes_to_documents(pdf_bytes, source_name=file_name)
            all_chunks.extend(chunks)
            ingested.append(file_name)
        except Exception as e:
            logging.warning(f"⚠️ '{file_name}' dizinlenemedi: {e}")
            failed.append((file_name, str(e)))

    if not all_chunks:
        # Tek bir kullanılabilir parça bile yoksa depo oluşturmanın anlamı yok.
        reasons = "; ".join(reason for _, reason in failed) or "bilinmeyen sebep"
        raise ValueError(f"Hiçbir dosya dizinlenemedi. Sebep: {reasons}")

    if embeddings is None:
        embeddings = _get_embeddings()

    # Aynı yola yeniden yazma ihtimaline karşı: eski depo tamamen silinir.
    # (app.py her denemede benzersiz dizin verdiği için normalde no-op'tur.)
    remove_tenant_vector_store(persist_directory)

    try:
        Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=TENANT_COLLECTION_NAME,
        )
    except Exception:
        # Yarım kalmış depoyu diskte bırakma.
        remove_tenant_vector_store(persist_directory)
        raise

    _write_manifest(persist_directory, ingested, len(all_chunks))

    logging.info(
        f"✅ Upload-time vectorization tamam: {len(ingested)} dosya, "
        f"{len(all_chunks)} parça kalıcı depoya yazıldı."
        + (f" Başarısız: {len(failed)}." if failed else "")
    )
    return {"chunk_count": len(all_chunks), "ingested": ingested, "failed": failed}


def ingest_pdf_to_persistent_store(
    pdf_bytes: bytes,
    persist_directory: str,
    embeddings: Optional[Embeddings] = None,
    source_name: str = "Yüklenen PDF",
) -> int:
    """
    Tek dosyalık geriye dönük uyumlu sarmalayıcı (çok dosyalı API'yi çağırır).

    Returns:
        Dizinlenen parça sayısı.
    """
    result = ingest_pdfs_to_persistent_store(
        [(source_name, pdf_bytes)], persist_directory, embeddings=embeddings
    )
    return result["chunk_count"]


def _connect_persistent_store(persist_directory: str, embeddings: Embeddings) -> Chroma:
    """
    Mevcut kalıcı depoya BAĞLANIR — doküman embed ETMEZ. Embedding
    fonksiyonu yalnızca sorgu metnini vektörlemek için gereklidir.
    """
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=TENANT_COLLECTION_NAME,
    )


def _resolve_top_k(persist_directory: str) -> int:
    """
    Depodaki dosya sayısına göre getirilecek parça sayısını uyarlar.

    Gerekçe: Tek dokümanda k=3 yeterliyken, 5 dokümanlık bir depoda aynı
    k, doğru pasajın hiç getirilmemesine yol açabilir — bu da Sözcü'nün
    ya "Veri bulunamadı" demesine ya da (grounding zayıfsa) uydurmasına
    zemin hazırlar. Dosya başına ~TOP_K_RESULTS parça hedeflenir,
    MAX_TOP_K_RESULTS ile sınırlanır (prompt boyutu kontrol altında kalsın).
    """
    file_count = len(get_tenant_manifest(persist_directory).get("files", []))
    if file_count <= 1:
        return TOP_K_RESULTS
    return min(TOP_K_RESULTS * file_count, MAX_TOP_K_RESULTS)


# ======================================================================
# 6. LANGGRAPH DÜĞÜMÜ: POLİTİKA/DOKÜMAN GETİRİCİ
# ======================================================================
def retrieve_policy_node(state: dict) -> dict:
    """
    LangGraph düğümü: soruya en alakalı doküman parçalarını getirir.

    Mod seçimi:
      - state["tenant_vector_store_path"] doluysa -> TENANT MODU:
        kiracının KALICI deposuna bağlan (yeniden embedding YOK) ve ara.
      - Anahtar yok/None ise -> VARSAYILAN MOD: dummy politika store'u.

    Bilinçli tasarım kararı — kayıp depo SESSİZCE varsayılana DÜŞMEZ:
      Anahtar dolu ama dizin diskte yoksa (GC süpürmüş / sunucu taşınmış),
      kullanıcı KENDİ dokümanını sorguladığını sanırken varsayılan demo
      politikalardan cevap almak bir veri bütünlüğü hatası olurdu. Bu
      durumda net, eyleme dönük bir hata döndürülür ("yeniden yükleyin").

    Sözleşme (Contract) — AgentState ile birebir uyumlu:
        Girdi : state["question"], opsiyonel state["tenant_vector_store_path"]
        Çıktı : {"result": <bağlam metni>} veya {"result": "HATA: ..."}
    """
    logging.info("📖 Düğüm: 'retrieve_policy_node' çalışıyor...")

    question = state.get("question", "").strip()
    if not question:
        return {"result": "HATA: Politika araması için geçerli bir soru bulunamadı."}

    tenant_store_path = state.get("tenant_vector_store_path") or None

    try:
        # --- TENANT MODU (bağlan, embed etme) ---
        if tenant_store_path:
            if not os.path.isdir(tenant_store_path):
                logging.warning(f"⚠️ Tenant deposu diskte yok: {tenant_store_path}")
                return {
                    "result": (
                        "HATA: Yüklediğiniz dokümanların dizini bulunamadı (oturum süresi "
                        "dolmuş olabilir). Lütfen dosyaları yeniden yükleyin."
                    )
                }

            top_k = _resolve_top_k(tenant_store_path)
            logging.info(f"🏢 Tenant modu: kalıcı depoya bağlanılıyor (k={top_k}, yeniden embedding YOK).")

            tenant_store = _connect_persistent_store(tenant_store_path, _get_embeddings())
            retrieved_docs = tenant_store.similarity_search(question, k=top_k)

            if not retrieved_docs:
                return {"result": "Yüklediğiniz dokümanlarda bu soruyla ilgili bir bölüm bulunamadı."}

            context_blocks: list[str] = []
            for i, doc in enumerate(retrieved_docs, start=1):
                source_file = doc.metadata.get("source_file", "Yüklenen PDF")
                page_no = doc.metadata.get("page")
                source_label = (
                    f"{source_file}, Sayfa {page_no + 1}"
                    if isinstance(page_no, int)
                    else source_file
                )
                context_blocks.append(f"[Doküman {i} | Kaynak: {source_label}]\n{doc.page_content}")

            logging.info(f"✅ RAG (tenant): {len(retrieved_docs)} alakalı bölüm getirildi.")
            return {"result": "\n\n".join(context_blocks)}

        # --- VARSAYILAN MOD (Phase 2 — değişmedi) ---
        vector_store = get_vector_store()
        retrieved_docs = vector_store.similarity_search(question, k=TOP_K_RESULTS)

        if not retrieved_docs:
            return {
                "result": (
                    "Kurumsal doküman arşivinde bu soruyla ilgili bir politika "
                    "bulunamadı."
                )
            }

        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("kaynak", "bilinmiyor")
            context_blocks.append(f"[Doküman {i} | Kaynak: {source}]\n{doc.page_content}")

        logging.info(f"✅ RAG: {len(retrieved_docs)} alakalı politika dokümanı getirildi.")
        return {"result": "\n\n".join(context_blocks)}

    except Exception as e:
        logging.error(f"RAG araması sırasında kritik hata: {e}")
        return {"result": f"HATA: Politika araması başarısız oldu. Detay: {e}"}


if __name__ == "__main__":
    # Sağlamlık testi (gerçek OpenAI embedding çağrısı yapar — OPENAI_API_KEY gerekir).
    test_question = "Kargo ücreti ne kadar? 500 TL üzeri alışverişte ücretsiz mi?"

    print("\n" + "=" * 60)
    print("🚀 RAG MODÜLÜ (rag_node.py) TEST EDİLİYOR — VARSAYILAN MOD")
    print("=" * 60)

    test_state = {"question": test_question, "query": "", "result": "", "error_count": 0}
    output = retrieve_policy_node(test_state)

    print("\n📖 GETİRİLEN KURUMSAL BAĞLAM:")
    print("-" * 60)
    print(output["result"])
    print("-" * 60)