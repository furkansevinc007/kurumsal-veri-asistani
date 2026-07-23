import logging
import os
import sqlite3
from typing import Any, Optional

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.pool import NullPool
from langchain_community.utilities import SQLDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_URI = "sqlite:///company.db"

# Bu sayıdan fazla tablo bulunursa uyarı loglanır: şema bağlamı prompt'a
# gömüldüğü için çok tablolu canlı veritabanları hem maliyeti hem gecikmeyi
# ciddi biçimde artırır (bkz. Phase 6 notları).
LARGE_SCHEMA_WARNING_THRESHOLD: int = 25

# Sürücüsü eksik olabilecek yaygın dialect'ler için kurulum ipuçları.
_DRIVER_HINTS: dict[str, str] = {
    "postgresql": "pip install psycopg2-binary",
    "mysql": "pip install pymysql",
    "mariadb": "pip install pymysql",
    "mssql": "pip install pyodbc",
    "oracle": "pip install oracledb",
}


# ======================================================================
# GÜVENLİK: BAĞLANTI DİZESİ MASKELEME
# ======================================================================
def redact_db_url(db_url: Optional[str]) -> str:
    """
    Bağlantı dizesini loglanabilir/gösterilebilir hale getirir (parola maskeli).

    Canlı DB özelliğiyle birlikte artık sisteme KİMLİK BİLGİSİ giriyor.
    Ham bağlantı dizesi ASLA loglanmamalı, debug panelinde gösterilmemeli
    veya hata mesajına konmamalıdır. Tüm bu noktalarda bu fonksiyon kullanılır.
    """
    if not db_url:
        return "(yok)"
    try:
        return make_url(db_url).render_as_string(hide_password=True)
    except Exception:
        # Ayrıştırılamayan dize: içeriğini sızdırmaktansa tamamen gizle.
        return "(geçersiz bağlantı dizesi)"


# 🚨 ZIRH 1: SQLite bağlantılarında Foreign Key kontrolünü zorla aktif eder.
#
# PHASE 6 KRİTİK DÜZELTMESİ: Bu dinleyici Engine SINIFI üzerinde kayıtlıdır,
# yani süreçteki HER engine'in her bağlantısında tetiklenir. Canlı DB URL
# desteğiyle birlikte bu, PostgreSQL/MySQL bağlantılarına da
# "PRAGMA foreign_keys=ON" göndermek anlamına gelirdi — bu ifade o
# veritabanlarında sözdizimi hatasıdır ve BAĞLANTIYI ANINDA KIRARDI.
# Çözüm: yalnızca gerçek sqlite3 bağlantılarında çalış.
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return  # SQLite değil -> PRAGMA gönderme.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


try:
    # 🚨 ZIRH 2: check_same_thread=False ile yapay zekanın asenkron çalışırken çökmütesini engelliyoruz
    engine = create_engine(DB_URI, connect_args={"check_same_thread": False})


    def get_sql_database() -> SQLDatabase:
        """
        LangChain için güvenli veritabanı objesi döndürür (varsayılan DB).

        NOT: LangChain ekosistemiyle uyumluluk için korunuyor; Text-to-SQL
        prompt bağlamı get_enriched_schema_context()'ten gelir.
        """
        return SQLDatabase(engine)

except Exception as e:
    logging.error(f"❌ Veritabanı Motoru Başlatılamadı! Detay: {e}")
    raise


# ======================================================================
# TENANT ENGINE YÖNETİMİ (Dosya tabanlı VE canlı URL tabanlı)
# ======================================================================
def _create_tenant_engine(tenant_db_path: str) -> Engine:
    """
    Kiracının yüklediği SQLite DOSYASI için kısa ömürlü bir engine üretir.

    - NullPool: bağlantı havuzu YOK; her kullanım sonrası dosya tanıtıcısı
      kapanır, geçici dosyalar silinirken kilit sorunu yaşanmaz.
    - Çağıran taraf engine.dispose() ile yaşam döngüsünü kapatmakla
      yükümlüdür (aşağıdaki fonksiyonlarda finally ile garanti edilir).

    Raises:
        FileNotFoundError: Dosya diskte yoksa. Sessizce varsayılan DB'ye
            DÜŞMEYİZ — kiracı verisi beklerken şirket verisi döndürmek
            çapraz kiracı sızıntısı olurdu.
    """
    if not os.path.isfile(tenant_db_path):
        raise FileNotFoundError(
            f"Kiracı veritabanı dosyası bulunamadı: {tenant_db_path}. "
            "Lütfen dosyayı yeniden yükleyin."
        )
    return create_engine(
        f"sqlite:///{tenant_db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )


def create_engine_from_url(tenant_db_url: str) -> Engine:
    """
    CANLI veritabanı bağlantı dizesinden kısa ömürlü bir engine üretir.

    Dikkat edilen noktalar:
      - connect_args yalnızca SQLite için verilir. `check_same_thread`
        SQLite'a özgüdür; PostgreSQL/MySQL sürücülerine gönderilirse
        bağlantı TypeError ile patlar.
      - NullPool: engine kısa ömürlüdür; havuzu süreçte tutmak, kimlik
        bilgisi taşıyan açık bağlantıları gereksiz yere yaşatırdı.
      - Hata mesajlarında bağlantı dizesi MASKELENİR (parola sızdırmaz).

    Raises:
        ValueError: Dize ayrıştırılamazsa veya gerekli sürücü kurulu değilse
            (kullanıcıya hangi paketi kuracağını söyleyen net mesajla).
    """
    try:
        url_obj = make_url(tenant_db_url)
    except Exception:
        raise ValueError(
            "Bağlantı dizesi geçersiz. Beklenen biçim: "
            "postgresql://kullanici:parola@sunucu:5432/veritabani"
        )

    backend = url_obj.get_backend_name()
    connect_args = {"check_same_thread": False} if backend == "sqlite" else {}

    try:
        return create_engine(tenant_db_url, connect_args=connect_args, poolclass=NullPool)
    except ModuleNotFoundError:
        hint = _DRIVER_HINTS.get(backend, f"'{backend}' sürücüsünü kurun")
        raise ValueError(
            f"'{backend}' veritabanı için gerekli Python sürücüsü kurulu değil. "
            f"Kurulum: {hint}"
        )
    except Exception as e:
        raise ValueError(f"Veritabanı motoru oluşturulamadı: {type(e).__name__}")


def get_active_dialect(
    tenant_db_url: Optional[str] = None,
    tenant_db_path: Optional[str] = None,
) -> str:
    """
    Aktif veri kaynağının SQL diyalektini döndürür ("sqlite", "postgresql"...).

    sql_generator.py bunu kullanarak prompt'taki diyalekt kurallarını
    dinamik seçer. Canlı bir PostgreSQL bağlıyken LLM'e "SQLite sözdizimi
    kullan" demek, üretilen her sorgunun patlaması demektir.

    Engine KURMAZ — yalnızca dizeyi ayrıştırır (ucuz ve yan etkisiz).
    """
    if tenant_db_url:
        try:
            return make_url(tenant_db_url).get_backend_name()
        except Exception:
            return "sqlite"
    # Dosya tabanlı kiracı DB'si ve varsayılan şirket DB'si SQLite'tır.
    return "sqlite"


def _resolve_engine(
    tenant_db_url: Optional[str],
    tenant_db_path: Optional[str],
    db_engine: Optional[Engine] = None,
) -> tuple[Engine, bool]:
    """
    Aktif engine'i ÖNCELİK SIRASINA göre seçer ve yaşam döngüsü sahipliğini
    bildirir.

    Öncelik (Phase 6 sözleşmesi):
        1. db_engine       -> doğrudan verilen engine (test enjeksiyonu)
        2. tenant_db_url   -> CANLI veritabanı (en yüksek kullanıcı önceliği)
        3. tenant_db_path  -> yüklenen SQLite dosyası
        4. (hiçbiri)       -> varsayılan şirket veritabanı

    Returns:
        (engine, owns_engine). owns_engine True ise çağıran, işi bitince
        engine.dispose() ÇAĞIRMAK ZORUNDADIR.
    """
    if db_engine is not None:
        return db_engine, False

    if tenant_db_url:
        logging.info(f"🔗 Canlı veritabanı kullanılıyor: {redact_db_url(tenant_db_url)}")
        return create_engine_from_url(tenant_db_url), True

    if tenant_db_path:
        logging.info("🏢 Yüklenen SQLite dosyası kullanılıyor.")
        return _create_tenant_engine(tenant_db_path), True

    return engine, False


def test_connection(tenant_db_url: str) -> str:
    """
    Bağlantı dizesini DOĞRULAR: engine kurar, gerçekten bağlanır ve kapatır.

    app.py bunu "Bağlan ve Doğrula" akışında kullanır; böylece kullanıcı
    hatayı ilk sorusunu sorduğunda değil, bağlanma anında görür.

    Returns:
        Başarılıysa maskelenmiş bağlantı özeti.

    Raises:
        ValueError: Sürücü eksik, dize geçersiz veya bağlantı kurulamadıysa
            (mesaj kimlik bilgisi İÇERMEZ).
    """
    active_engine = create_engine_from_url(tenant_db_url)
    try:
        with active_engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return redact_db_url(tenant_db_url)
    except Exception as e:
        # Sürücü hataları bağlantı dizesini mesajına gömebilir -> yalnızca
        # istisna TÜRÜNÜ sızdır, metnini değil.
        raise ValueError(
            f"Veritabanına bağlanılamadı ({type(e).__name__}). "
            "Sunucu adresini, kimlik bilgilerini ve ağ erişimini kontrol edin."
        )
    finally:
        active_engine.dispose()


# ======================================================================
# ZENGİNLEŞTİRİLMİŞ ŞEMA ÇIKARIMI (Production-Grade Schema Extraction)
# ======================================================================
def get_enriched_schema_context(
    sample_row_limit: int = 3,
    db_engine: Optional[Engine] = None,
    tenant_db_path: Optional[str] = None,
    tenant_db_url: Optional[str] = None,
) -> str:
    """
    LLM'e enjekte edilecek zenginleştirilmiş şema bağlamını üretir:
    tablo adları, kolonlar + veri tipleri, PK/NOT NULL işaretleri,
    Foreign Key ilişkileri ve her tablodan örnek veri satırları.

    Kaynak önceliği: db_engine > tenant_db_url > tenant_db_path > varsayılan.

    ⚠️ CANLI VERİTABANI NOTU: Örnek satırlar GERÇEK üretim verisi olabilir
    ve prompt ile birlikte LLM sağlayıcısına gider. Kişisel veri içeren
    kurulumlarda (KVKK/GDPR) sample_row_limit=0 verilerek örnekleme
    tamamen kapatılabilir; bu durumda yalnızca yapısal şema gönderilir.

    Args:
        sample_row_limit: Tablo başına örnek satır sayısı. 0 veya negatifse
            örnekleme YAPILMAZ (gizlilik modu).
        db_engine: Test enjeksiyonu için opsiyonel engine override'ı.
        tenant_db_path: Kiracının yüklediği SQLite dosyasının yolu.
        tenant_db_url: Canlı veritabanı bağlantı dizesi (en yüksek öncelik).

    Returns:
        Prompt'a doğrudan enjekte edilebilecek formatlı tam şema metni.
    """
    active_engine, owns_engine = _resolve_engine(tenant_db_url, tenant_db_path, db_engine)

    try:
        inspector = inspect(active_engine)
        table_names = inspector.get_table_names()

        if not table_names:
            logging.warning("⚠️ Şema çıkarımı: veritabanında hiçbir tablo bulunamadı.")
            return "⚠️ Veritabanında hiçbir tablo bulunamadı."

        if len(table_names) > LARGE_SCHEMA_WARNING_THRESHOLD:
            logging.warning(
                f"⚠️ Şema çıkarımı: {len(table_names)} tablo bulundu. Büyük şemalar "
                "prompt boyutunu, maliyeti ve gecikmeyi ciddi biçimde artırır."
            )

        schema_blocks: list[str] = []

        for table_name in table_names:
            block_lines: list[str] = [f"### Tablo: {table_name}"]

            # --- 1. Kolonlar ve Veri Tipleri ---
            columns = inspector.get_columns(table_name)
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_columns = set(pk_constraint.get("constrained_columns") or [])

            block_lines.append("Kolonlar:")
            for col in columns:
                col_name = col["name"]
                col_type = str(col["type"])
                markers = []
                if col_name in pk_columns:
                    markers.append("PRIMARY KEY")
                if not col.get("nullable", True):
                    markers.append("NOT NULL")
                marker_str = f" [{', '.join(markers)}]" if markers else ""
                block_lines.append(f"  - {col_name} ({col_type}){marker_str}")

            # --- 2. Foreign Key İlişkileri ---
            foreign_keys = inspector.get_foreign_keys(table_name)
            if foreign_keys:
                block_lines.append("İlişkiler (Foreign Keys):")
                for fk in foreign_keys:
                    local_cols = ", ".join(fk["constrained_columns"])
                    remote_table = fk["referred_table"]
                    remote_cols = ", ".join(fk["referred_columns"])
                    block_lines.append(
                        f"  - {table_name}.{local_cols} -> {remote_table}.{remote_cols}"
                    )

            # --- 3. Örnek Veri Satırları ---
            if sample_row_limit <= 0:
                block_lines.append("Örnek Veriler: (gizlilik modu — örnekleme kapalı)")
            else:
                try:
                    with active_engine.connect() as connection:
                        sample_result = connection.execute(
                            text(f'SELECT * FROM "{table_name}" LIMIT :row_limit'),
                            {"row_limit": sample_row_limit},
                        )
                        sample_rows = sample_result.fetchall()
                        sample_columns = list(sample_result.keys())

                    if sample_rows:
                        block_lines.append(f"Örnek Veriler (İlk {len(sample_rows)} Satır):")
                        block_lines.append(f"  {' | '.join(sample_columns)}")
                        for row in sample_rows:
                            block_lines.append(f"  {' | '.join(str(value) for value in row)}")
                    else:
                        block_lines.append("Örnek Veriler: (tablo şu anda boş)")

                except Exception as e:
                    # Örnek veri okunamaması şema çıkarımını bloke etmemeli.
                    # (Bazı diyalektlerde LIMIT sözdizimi farklıdır; yapısal
                    # şema yine de LLM'e sunulur.)
                    logging.warning(f"⚠️ '{table_name}' tablosundan örnek veri okunamadı: {e}")
                    block_lines.append("Örnek Veriler: (okunamadı)")

            schema_blocks.append("\n".join(block_lines))

        return "\n\n".join(schema_blocks)

    finally:
        if owns_engine:
            active_engine.dispose()


# ======================================================================
# SQL ÇALIŞTIRMA (Tenant-Aware: canlı URL / yüklenen dosya / varsayılan)
# ======================================================================
def execute_sql_query(
    query: str,
    tenant_db_path: Optional[str] = None,
    tenant_db_url: Optional[str] = None,
) -> list[Any]:
    """
    Verilen SQL sorgusunu, öncelik sırasına göre seçilen veri kaynağında
    çalıştırır ve tüm satırları döndürür.

    Kaynak önceliği: tenant_db_url > tenant_db_path > varsayılan şirket DB.

    ÖNEMLİ GÜVENLİK NOTU: SELECT-only güvenlik zırhı agent.py'deki
    execute_sql_node'da uygulanır; bu fonksiyon o katmanın ALTINDA kalan
    ham erişim katmanıdır ve doğrudan dış girdilere maruz bırakılmamalıdır.

    Raises:
        FileNotFoundError: tenant_db_path verilmiş ama dosya yoksa.
        ValueError: Canlı bağlantı dizesi geçersiz/sürücü eksikse.
        Exception: SQL çalıştırma hataları olduğu gibi yukarı fırlatılır —
            hata sözleşmesini (HATA: öneki, error_count) agent.py yönetir.
    """
    active_engine, owns_engine = _resolve_engine(tenant_db_url, tenant_db_path)

    try:
        with active_engine.connect() as connection:
            result_proxy = connection.execute(text(query))
            return result_proxy.fetchall()
    finally:
        if owns_engine:
            active_engine.dispose()


if __name__ == "__main__":
    # Sistemin sağlamlık testi
    try:
        db = get_sql_database()
        tables = db.get_usable_table_names()
        if not tables:
            logging.warning("⚠️ Veritabanına bağlanıldı ama içinde tablo bulunamadı!")
        else:
            logging.info("✅ Veritabanı Motoru Aktif ve Kusursuz Çalışıyor!")
            logging.info(f"📦 Okunabilir Tablolar: {tables}")

            enriched = get_enriched_schema_context()
            logging.info("📐 Zenginleştirilmiş şema bağlamı (önizleme):\n" + enriched)
    except Exception as e:
        logging.error(f"❌ Test sırasında bağlantı çöktü: {e}")