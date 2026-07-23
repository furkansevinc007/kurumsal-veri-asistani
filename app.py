"""
app.py

Kurumsal Veri Asistanı — Streamlit arayüzü (Hybrid SaaS).

Bu katman SADECE sunum ve orkestrasyon yapar; ağır işler alan
modüllerindedir:
  - PDF -> vektör dönüşümü, kalıcı depo ve GC : rag_node.py
  - Bağlantı doğrulama, şema, sorgu çalıştırma : database.py
  - Sohbet akışı sözleşmesi                    : chat_engine.py
  - Araç seçimi (tool calling) döngüsü         : graph.py

VERİ KAYNAĞI ÖNCELİĞİ (tüm katmanlarda aynı):
    Canlı DB URL  >  Yüklenen .db dosyası  >  Varsayılan demo verisi

YÜKLEME DAYANIKLILIĞI:
  1. HER yükleme denemesi BENZERSİZ bir depo dizinine yazar -> dosya
     kilidi ve çapraz doküman kirlenmesi matematiksel olarak imkânsız.
  2. Kaldırma işlemi TÜM ilgili durum anahtarlarını KOŞULSUZ sıfırlar.
  3. Başarısız işleme ne tuzak kurar ne de her rerun'da pahalı işlemi
     yeniden dener (imza kaydedilir + açık "Tekrar Dene" düğmesi).

Çalıştırma:
    streamlit run app.py
"""

# ======================================================================
# 🚨 STREAMLIT COMMUNITY CLOUD — SQLITE SÜRÜM YAMASI
# ======================================================================
# Streamlit Cloud'un Debian imajı eski bir sistem SQLite'ı ile gelir.
# ChromaDB ise daha yeni bir sürüm ister ve aksi hâlde daha ilk
# başlatmada çöker. Aşağıdaki blok, standart kütüphanenin `sqlite3`
# modülünü paketlenmiş güncel `pysqlite3` ile değiştirir.
#
# ⚠️ KONUM KRİTİKTİR: Bu blok, sqlite3'e dokunan HERHANGİ bir importtan
# (chromadb, langchain_chroma, sqlalchemy, database.py, rag_node.py)
# ÖNCE çalışmalıdır. Bir modül sqlite3'ü bir kez içe aktardıktan sonra
# takas etmek geç kalmış olur. Bu yüzden dosyanın en üstündedir —
# yalnızca modül docstring'i öndedir ve o hiçbir şey import etmez.
#
# Yerelde pysqlite3 kurulu değilse ImportError sessizce yutulur ve
# sistem SQLite'ı kullanılmaya devam eder (geliştirme ortamı bozulmaz).
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import uuid

import streamlit as st

from chat_engine import stream_chat_response
from database import redact_db_url, test_connection
from rag_node import (
    cleanup_stale_tenant_data,
    get_tenant_manifest,
    get_tenant_session_dir,
    ingest_pdfs_to_persistent_store,
    remove_tenant_vector_store,
    touch_tenant_dir,
)

# Dosya sihirli baytları (magic bytes) — içerik doğrulama, uzantıya güven yok.
_SQLITE_MAGIC = b"SQLite format 3\x00"
_PDF_MAGIC = b"%PDF"


# ======================================================================
# 1. SAYFA, OTURUM VE AÇILIŞ TEMİZLİĞİ
# ======================================================================
st.set_page_config(
    page_title="Kurumsal Veri Asistanı",
    page_icon="🤖",
    layout="centered",
)


def _init_session_state() -> None:
    """Oturum değişkenlerini (ilk açılışta) güvenli şekilde başlatır."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "data_session_id" not in st.session_state:
        # Veri oturumu kimliği thread_id'den AYRIDIR: "Sohbeti Temizle"
        # thread_id'yi döndürür (hafıza sıfırlanır) ama yüklenen veriler
        # bu kimliğe bağlı dizinde yaşamaya devam eder.
        st.session_state.data_session_id = str(uuid.uuid4())

    # --- Yerel dosya kaynakları ---
    st.session_state.setdefault("tenant_db_path", None)
    st.session_state.setdefault("tenant_db_sig", None)
    st.session_state.setdefault("tenant_db_error", None)
    st.session_state.setdefault("tenant_vector_store_path", None)
    st.session_state.setdefault("tenant_pdf_sig", None)
    st.session_state.setdefault("tenant_pdf_error", None)
    st.session_state.setdefault("tenant_pdf_warning", None)

    # --- Canlı DB kaynağı (Phase 6) ---
    st.session_state.setdefault("tenant_db_url", None)        # DOĞRULANMIŞ bağlantı dizesi
    st.session_state.setdefault("tenant_db_url_label", None)  # maskeli gösterim
    st.session_state.setdefault("tenant_db_url_error", None)

    if "gc_done" not in st.session_state:
        # Açılış Çöp Toplayıcısı: oturum başına BİR kez koşar.
        deleted = cleanup_stale_tenant_data()
        if deleted:
            st.toast(f"{deleted} eski oturuma ait geçici veri temizlendi.", icon="🧹")
        st.session_state.gc_done = True


_init_session_state()

# Aktif oturumun dizinine her etkileşimde dokun -> GC eşiği "son
# kullanımdan beri" işler; yaşayan oturum asla süpürülmez.
_session_dir = get_tenant_session_dir(st.session_state.data_session_id)
touch_tenant_dir(_session_dir)


def _safe_unlink(path: str | None) -> None:
    """Tek bir dosyayı sessiz-güvenli siler (yoksa/kilitliyse geçer)."""
    try:
        if path and os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass  # En kötü ihtimalle açılış GC'si süpürür.


def _reset_pdf_state() -> None:
    """
    PDF ile ilgili TÜM oturum durumunu koşulsuz sıfırlar ve varsa aktif
    vektör deposunu diskten siler.

    Tek yerde toplanmasının nedeni: kaldırma, hata ve yeniden deneme
    yollarının üçü de aynı anahtar kümesini temizlemek zorundadır. Biri
    unutulursa kullanıcı tutarsız bir ara duruma sıkışır.
    """
    if st.session_state.tenant_vector_store_path:
        remove_tenant_vector_store(st.session_state.tenant_vector_store_path)
    st.session_state.tenant_vector_store_path = None
    st.session_state.tenant_pdf_sig = None
    st.session_state.tenant_pdf_error = None
    st.session_state.tenant_pdf_warning = None


# ======================================================================
# 2. SIDEBAR
# ======================================================================
with st.sidebar:
    st.title("⚙️ Çalışma Alanı")

    debug_mode: bool = st.toggle(
        "🔬 Geliştirici Görünümü",
        value=False,
        help=(
            "Asistanın perde arkasını gösterir: hangi araçları kullandığı, "
            "hangi kaynaklara başvurduğu ve elde ettiği ham veriler."
        ),
    )

    if st.button("✨ Yeni Sohbet Başlat", use_container_width=True):
        # UI geçmişi sıfırlanır ve YENİ thread_id üretilir — backend'deki
        # (MemorySaver) konuşma hafızası da fiilen sıfırdan başlar.
        # Bağlı veri kaynakları bilinçli olarak KORUNUR.
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.subheader("📁 Veri Kaynaklarınız")

    tab_files, tab_live = st.tabs(["📂 Dosya Yükle", "🔗 Canlı Bağlantı"])

    # ==================================================================
    # SEKME 1 — DOSYALAR
    # ==================================================================
    with tab_files:
        st.caption(
            "Kendi veritabanınızı ve belgelerinizi yükleyin; asistan anında "
            "onlar üzerinden çalışmaya başlasın. Verileriniz yalnızca size ayrılmış "
            "izole bir alanda işlenir ve en geç 12 saat içinde otomatik olarak silinir."
        )

        # --- SQLite DB yükleyici (tek dosya) ---
        uploaded_db = st.file_uploader(
            "SQLite Veritabanı (.db)",
            type=["db", "sqlite", "sqlite3"],
            help="Yüklediğiniz anda tüm veri sorularınız bu veritabanından yanıtlanır.",
            key="db_uploader",
        )

        if uploaded_db is None:
            if (
                st.session_state.tenant_db_path
                or st.session_state.tenant_db_sig
                or st.session_state.tenant_db_error
            ):
                _safe_unlink(st.session_state.tenant_db_path)
                st.session_state.tenant_db_path = None
                st.session_state.tenant_db_sig = None
                st.session_state.tenant_db_error = None
                st.toast("Veritabanı çıkarıldı. Demo verisine dönüldü.", icon="↩️")
        else:
            # Streamlit rerun tuzağı: uploader, dosya widget'ta durduğu
            # sürece objeyi HER rerun'da döndürür. İmza kontrolü olmadan
            # her etkileşimde dosya yeniden yazılırdı.
            db_sig = (uploaded_db.name, uploaded_db.size)
            if st.session_state.tenant_db_sig != db_sig:
                st.session_state.tenant_db_error = None
                previous_db_path = st.session_state.tenant_db_path
                new_db_path = os.path.join(_session_dir, f"tenant_db_{uuid.uuid4().hex[:8]}.db")

                try:
                    db_bytes = uploaded_db.getvalue()

                    # İçerik doğrulama: uzantı değil, sihirli baytlar konuşur.
                    if not db_bytes.startswith(_SQLITE_MAGIC):
                        raise ValueError(
                            "Bu dosya bir SQLite veritabanı gibi görünmüyor. "
                            "Lütfen doğru dosyayı seçtiğinizden emin olun."
                        )

                    with st.spinner("Veritabanınız hazırlanıyor..."):
                        with open(new_db_path, "wb") as f:
                            f.write(db_bytes)

                    st.session_state.tenant_db_path = new_db_path
                    st.session_state.tenant_db_sig = db_sig
                    _safe_unlink(previous_db_path)
                    st.toast(
                        f"“{uploaded_db.name}” hazır. Artık veri sorularınızı sorabilirsiniz.",
                        icon="✅",
                    )

                except Exception as e:
                    _safe_unlink(new_db_path)
                    # İmzayı KAYDET: aynı bozuk dosya her rerun'da yeniden
                    # denenmesin (kaçış yolları aşağıda).
                    st.session_state.tenant_db_sig = db_sig
                    st.session_state.tenant_db_error = (
                        str(e)
                        if isinstance(e, ValueError)
                        else (
                            "Veritabanınız hazırlanırken beklenmedik bir sorun oluştu. "
                            "Lütfen yeniden deneyin."
                        )
                    )

        if st.session_state.tenant_db_error:
            st.error(st.session_state.tenant_db_error)
            if st.button("🔄 Yeniden Dene", key="db_retry", use_container_width=True):
                st.session_state.tenant_db_sig = None
                st.session_state.tenant_db_error = None
                st.rerun()

        st.markdown("---")

        # --- PDF yükleyici (ÇOKLU dosya — Phase 6) ---
        uploaded_pdfs = st.file_uploader(
            "Kurumsal Dokümanlar (.pdf)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Birden fazla belge seçebilirsiniz; hepsi birlikte aranır.",
            key="pdf_uploader",
        )

        if not uploaded_pdfs:
            # KOŞULSUZ sıfırlama: yol, imza, hata ve uyarı birlikte temizlenir.
            if (
                st.session_state.tenant_vector_store_path
                or st.session_state.tenant_pdf_sig
                or st.session_state.tenant_pdf_error
            ):
                _reset_pdf_state()
                st.toast("Belgeler çıkarıldı. Örnek politikalara dönüldü.", icon="↩️")
        else:
            # Çoklu dosya imzası: (ad, boyut) ikililerinin SIRALI demeti.
            # Sıralama önemli: kullanıcı aynı dosyaları farklı sırada
            # seçtiğinde gereksiz yere yeniden dizinleme yapılmasın.
            pdf_sig = tuple(sorted((f.name, f.size) for f in uploaded_pdfs))

            if st.session_state.tenant_pdf_sig != pdf_sig:
                st.session_state.tenant_pdf_error = None
                st.session_state.tenant_pdf_warning = None

                # HER denemeye BENZERSİZ dizin (kilit + kirlenme koruması).
                new_store_dir = os.path.join(_session_dir, f"vector_store_{uuid.uuid4().hex[:8]}")
                previous_store_dir = st.session_state.tenant_vector_store_path

                try:
                    # Geçerli PDF'leri ayıkla (sihirli bayt kontrolü).
                    valid_files: list[tuple[str, bytes]] = []
                    invalid_names: list[str] = []
                    for f in uploaded_pdfs:
                        data = f.getvalue()
                        if data.startswith(_PDF_MAGIC):
                            valid_files.append((f.name, data))
                        else:
                            invalid_names.append(f.name)

                    if not valid_files:
                        raise ValueError(
                            "Seçtiğiniz dosyaların hiçbiri okunabilir bir PDF değil. "
                            "Lütfen dosyalarınızı kontrol edip yeniden deneyin."
                        )

                    # AĞIR İŞ BURADA, BİR KEZ: tüm dosyalar tek depoya yazılır.
                    # Soru anında yalnızca bu depoya bağlanılır (yeniden
                    # embedding YOK -> sıfır bekleme, sıfır ek API maliyeti).
                    with st.spinner(
                        f"{len(valid_files)} belge okunuyor ve aranabilir hâle getiriliyor. "
                        "Bu işlem yalnızca bir kez yapılır; sonraki sorularınız anında yanıtlanır."
                    ):
                        result = ingest_pdfs_to_persistent_store(valid_files, new_store_dir)

                    st.session_state.tenant_vector_store_path = new_store_dir
                    st.session_state.tenant_pdf_sig = pdf_sig
                    if previous_store_dir and previous_store_dir != new_store_dir:
                        remove_tenant_vector_store(previous_store_dir)

                    # Kısmi başarı: dizinlenenler çalışır, başarısızlar bildirilir.
                    skipped = [name for name, _ in result["failed"]] + invalid_names
                    if skipped:
                        st.session_state.tenant_pdf_warning = (
                            "Şu belgeler okunamadı ve aramalara dâhil edilmedi: "
                            + ", ".join(skipped)
                            + ". Diğer belgeleriniz normal şekilde kullanılıyor."
                        )

                    st.toast(
                        f"{len(result['ingested'])} belge hazır. "
                        "Artık içerikleri hakkında soru sorabilirsiniz.",
                        icon="✅",
                    )

                except Exception as e:
                    # Hata kullanıcıyı TUZAĞA DÜŞÜRMEZ ve TEKRAR FIRTINASI yaratmaz.
                    remove_tenant_vector_store(new_store_dir)  # yarım depoyu bırakma
                    _reset_pdf_state()  # eski dokümanı da devre dışı bırak (yanlış kaynak riski)
                    st.session_state.tenant_pdf_sig = pdf_sig  # imzayı KAYDET -> tek deneme
                    st.session_state.tenant_pdf_error = (
                        str(e)
                        if isinstance(e, ValueError)
                        else (
                            "Belgeleriniz hazırlanırken beklenmedik bir sorun oluştu. "
                            "Lütfen yeniden deneyin veya farklı bir dosya yükleyin."
                        )
                    )

        if st.session_state.tenant_pdf_error:
            st.error(st.session_state.tenant_pdf_error)
            if st.button("🔄 Yeniden Dene", key="pdf_retry", use_container_width=True):
                st.session_state.tenant_pdf_sig = None
                st.session_state.tenant_pdf_error = None
                st.rerun()

        if st.session_state.tenant_pdf_warning:
            st.warning(st.session_state.tenant_pdf_warning)

        # Dizinlenmiş dosyaların listesi (şeffaflık).
        manifest = get_tenant_manifest(st.session_state.tenant_vector_store_path)
        if manifest["files"]:
            with st.expander(f"📑 Kullanıma hazır belgeler ({len(manifest['files'])})", expanded=False):
                for name in manifest["files"]:
                    st.write(f"• {name}")

    # ==================================================================
    # SEKME 2 — CANLI DB URL
    # ==================================================================
    with tab_live:
        st.caption(
            "Mevcut veritabanınıza doğrudan bağlanın; veri kopyalamanıza gerek yok. "
            "Bağlantı bilgileriniz yalnızca bu oturum boyunca saklanır. "
            "Güvenliğiniz için **salt okunur (read-only)** bir kullanıcı ile bağlanmanızı öneririz."
        )

        # type="password": bağlantı dizesi parola içerir, ekranda açık durmamalı.
        db_url_input = st.text_input(
            "Bağlantı Dizesi",
            type="password",
            placeholder="postgresql://kullanici:parola@sunucu:5432/veritabani",
            help="PostgreSQL, MySQL, MSSQL, Oracle ve SQLite desteklenir.",
            key="db_url_input",
        )

        col_connect, col_disconnect = st.columns(2)

        with col_connect:
            if st.button("🔌 Bağlan", use_container_width=True, key="db_url_connect"):
                candidate = (db_url_input or "").strip()
                if not candidate:
                    st.session_state.tenant_db_url_error = "Lütfen bir bağlantı dizesi girin."
                else:
                    try:
                        # Bağlantıyı ŞİMDİ doğrula: kullanıcı hatayı ilk
                        # sorusunu sorduğunda değil, burada görsün.
                        with st.spinner("Bağlantınız test ediliyor..."):
                            label = test_connection(candidate)
                        st.session_state.tenant_db_url = candidate
                        st.session_state.tenant_db_url_label = label
                        st.session_state.tenant_db_url_error = None
                        st.toast(
                            "Bağlantı başarılı. Sorularınız artık canlı verinizden yanıtlanacak.",
                            icon="✅",
                        )
                    except ValueError as ve:
                        st.session_state.tenant_db_url = None
                        st.session_state.tenant_db_url_label = None
                        st.session_state.tenant_db_url_error = str(ve)
                    except Exception:
                        st.session_state.tenant_db_url = None
                        st.session_state.tenant_db_url_label = None
                        st.session_state.tenant_db_url_error = (
                            "Bağlantı kurulurken beklenmedik bir sorun oluştu. "
                            "Lütfen bilgilerinizi kontrol edip yeniden deneyin."
                        )

        with col_disconnect:
            if st.button("⛔ Bağlantıyı Kes", use_container_width=True, key="db_url_disconnect"):
                st.session_state.tenant_db_url = None
                st.session_state.tenant_db_url_label = None
                st.session_state.tenant_db_url_error = None
                st.toast("Bağlantı sonlandırıldı.", icon="↩️")

        if st.session_state.tenant_db_url_error:
            st.error(st.session_state.tenant_db_url_error)

        if st.session_state.tenant_db_url:
            st.success(f"Bağlantı etkin · `{st.session_state.tenant_db_url_label}`")
            if st.session_state.tenant_db_path:
                # Öncelik kuralı kullanıcıya AÇIKÇA söylenir; sessiz
                # önceliklendirme "neden yüklediğim dosya kullanılmıyor?"
                # sorusuna yol açardı.
                st.info(
                    "ℹ️ Canlı bağlantınız öncelikli. Yüklediğiniz `.db` dosyası şu an "
                    "kullanılmıyor; dosyaya dönmek için bağlantıyı sonlandırmanız yeterli."
                )

    # ==================================================================
    # AKTİF KAYNAK GÖSTERGESİ (öncelik kuralını yansıtır)
    # ==================================================================
    st.divider()
    if st.session_state.tenant_db_url:
        sql_status = f"🔗 Canlı bağlantı · `{st.session_state.tenant_db_url_label}`"
    elif st.session_state.tenant_db_path:
        sql_status = "🟢 Yüklediğiniz veritabanı"
    else:
        sql_status = "⚪ Örnek demo verisi"

    doc_manifest = get_tenant_manifest(st.session_state.tenant_vector_store_path)
    if doc_manifest["files"]:
        doc_status = f"🟢 {len(doc_manifest['files'])} belgeniz"
    else:
        doc_status = "⚪ Örnek politika metinleri"

    st.caption(f"**Şu an kullanılan kaynaklar**\n\nVeri: {sql_status}\n\nBelgeler: {doc_status}")
    st.caption(f"Oturum kimliği · `{st.session_state.thread_id[:8]}…`")


# ======================================================================
# 3. ANA UI: BAŞLIK + GEÇMİŞ MESAJLAR
# ======================================================================
st.title("🤖 Kurumsal Veri Asistanı")
st.caption("Satış verilerinizden İK politikalarınıza kadar tüm kurumsal bilginize saniyeler içinde ulaşın. Asistan, karmaşık sorularınızda gerektiğinde hem veritabanınızı hem de dokümanlarınızı aynı anda tarayarak size en doğru ve bütüncül cevabı sunar.")


def _render_debug_panel(meta: dict) -> None:
    """
    Debug detaylarını gösterir (bağlantı dizesi DAİMA maskelidir).

    Phase 7: Motor artık tool-calling olduğu için panel, hangi ARAÇLARIN
    çağrıldığını da gösterir. Üretilen SQL sorgusu araç çıktısının içinde
    yer aldığından ayrı bir alan yerine ham sonuçta görünür.
    """
    with st.expander("🔬 Bu cevap nasıl oluşturuldu?", expanded=False):
        tools_used = meta.get("tools_used") or []
        st.markdown(
            f"**Başvurulan kaynaklar:** {', '.join(tools_used) if tools_used else 'Hiçbir veri kaynağı kullanılmadı'}"
        )

        sources = meta.get("sources") or []
        if sources:
            st.markdown("**Kaynaklar:** " + ", ".join(sources))

        if meta.get("tenant_db_url"):
            st.markdown(f"**Canlı bağlantı:** `{meta['tenant_db_url']}`")

        st.text(f"Ham veri çıktısı:\n{meta.get('raw_result', '-')}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if debug_mode and message.get("meta"):
            _render_debug_panel(message["meta"])


# ======================================================================
# 4. YENİ SORU AKIŞI
# ======================================================================
if user_question := st.chat_input(
    "Verileriniz hakkında bir soru sorun…"
):
    st.session_state.messages.append({"role": "user", "content": user_question, "meta": None})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        status_box = st.status("Sorunuz hazırlanıyor...", expanded=debug_mode)
        answer_placeholder = st.empty()

        accumulated_answer: str = ""
        response_meta: dict | None = None
        had_fatal_error: bool = False

        try:
            for event in stream_chat_response(
                user_question,
                st.session_state.thread_id,
                tenant_db_path=st.session_state.tenant_db_path,
                tenant_db_url=st.session_state.tenant_db_url,
                tenant_vector_store_path=st.session_state.tenant_vector_store_path,
            ):
                event_type = event.get("type")

                if event_type == "status":
                    status_box.update(label=event["message"])
                    if debug_mode:
                        status_box.write(event["message"])

                elif event_type == "meta":
                    response_meta = event

                elif event_type == "token":
                    accumulated_answer += event.get("content", "")
                    answer_placeholder.markdown(accumulated_answer + "▌")

                elif event_type == "replace":
                    # Sözcü kurala rağmen metin içine atıf serpiştirdiyse
                    # servis katmanı temizlenmiş metni gönderir; ekranda
                    # gösterilen sürümü onunla değiştiriyoruz.
                    accumulated_answer = event.get("content", accumulated_answer)
                    answer_placeholder.markdown(accumulated_answer + "▌")

                elif event_type == "error":
                    had_fatal_error = True
                    accumulated_answer = event.get(
                        "message",
                        "Üzgünüm, bu isteği şu anda tamamlayamadım. Lütfen yeniden deneyin.",
                    )
                    answer_placeholder.markdown(accumulated_answer)

        except Exception:
            had_fatal_error = True
            accumulated_answer = (
                "Üzgünüm, beklenmedik bir sorun oluştu ve sorunuzu tamamlayamadım. "
                "Lütfen birazdan yeniden deneyin."
            )
            answer_placeholder.markdown(accumulated_answer)

        answer_placeholder.markdown(accumulated_answer)

        if had_fatal_error:
            status_box.update(label="Bu isteği tamamlayamadık", state="error", expanded=False)
        else:
            status_box.update(label="Cevabınız hazır", state="complete", expanded=False)

        if debug_mode and response_meta:
            _render_debug_panel(response_meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": accumulated_answer, "meta": response_meta}
    )