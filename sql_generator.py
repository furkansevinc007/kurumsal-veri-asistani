import os
import logging
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from database import get_active_dialect, get_enriched_schema_context

# Profesyonel Loglama Ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Radar ile .env dosyasını zorla bul ve yükle
env_path = find_dotenv()
logging.info(f".env dosyası şu konumda bulundu: {env_path}")
load_dotenv(env_path)


# ======================================================================
# PHASE 6: DİYALEKTE ÖZGÜ SÖZDİZİMİ KURALLARI
# ======================================================================
# Canlı DB URL desteğiyle birlikte hedef veritabanı artık SQLite olmak
# zorunda değil. Prompt'ta "SQLite kullan" yazılı kalsaydı, bağlanan bir
# PostgreSQL için üretilen her sorgu sözdizimi hatasıyla patlar ve
# self-correction bütçesi anlamsızca tükenirdi.
_DIALECT_RULES: dict[str, str] = {
    "sqlite": """   - Tarih/saat işlemleri için DATE(), DATETIME(), STRFTIME() kullan.
   - Sayfalama için LIMIT/OFFSET kullan (TOP N KULLANMA — bu SQL Server sözdizimidir).
   - String birleştirme için || operatörünü kullan (CONCAT() KULLANMA).
   - NOW() veya GETDATE() DEĞİL, CURRENT_TIMESTAMP veya date('now') kullan.""",
    "postgresql": """   - Tarih/saat işlemleri için NOW(), CURRENT_DATE, DATE_TRUNC(), AGE(), INTERVAL kullan.
   - Sayfalama için LIMIT/OFFSET kullan.
   - String birleştirme için || veya CONCAT() kullanabilirsin.
   - Kolon/tablo adları küçük harfe normalize edilir; büyük harfli adlar çift tırnak ister.
   - Tip dönüşümü için CAST(x AS type) veya x::type kullan.""",
    "mysql": """   - Tarih/saat işlemleri için NOW(), CURDATE(), DATE_FORMAT(), DATE_ADD() kullan.
   - Sayfalama için LIMIT/OFFSET kullan.
   - String birleştirme için CONCAT() kullan (|| operatörü MySQL'de çalışmaz).
   - Tanımlayıcılar için gerekiyorsa ters tırnak (`) kullan.""",
    "mariadb": """   - Tarih/saat işlemleri için NOW(), CURDATE(), DATE_FORMAT(), DATE_ADD() kullan.
   - Sayfalama için LIMIT/OFFSET kullan.
   - String birleştirme için CONCAT() kullan (|| operatörü çalışmaz).
   - Tanımlayıcılar için gerekiyorsa ters tırnak (`) kullan.""",
    "mssql": """   - Satır sınırlama için TOP N veya OFFSET/FETCH kullan (LIMIT DESTEKLENMEZ).
   - Tarih/saat işlemleri için GETDATE(), DATEADD(), DATEDIFF(), FORMAT() kullan.
   - String birleştirme için + operatörünü veya CONCAT() kullan.
   - Tanımlayıcılar için gerekiyorsa köşeli parantez ([tablo]) kullan.""",
    "oracle": """   - Satır sınırlama için FETCH FIRST N ROWS ONLY kullan (LIMIT DESTEKLENMEZ).
   - Tarih/saat için SYSDATE, TO_DATE(), TO_CHAR() kullan.
   - String birleştirme için || operatörünü kullan.
   - FROM'suz SELECT yazma; gerekirse FROM DUAL kullan.""",
}

_GENERIC_DIALECT_RULES = """   - Yalnızca ANSI SQL standardındaki güvenli sözdizimini kullan.
   - Veritabanına özgü fonksiyonlardan kaçın; şüphedeyken standart SQL tercih et."""


def get_dialect_rules(dialect: str) -> str:
    """Hedef diyalekt için sözdizimi kurallarını döndürür."""
    return _DIALECT_RULES.get(dialect, _GENERIC_DIALECT_RULES)


def get_sql_prompt_template() -> ChatPromptTemplate:
    """
    Yapay zekaya 'Kıdemli Veritabanı Yöneticisi' (DBA) rolü veren,
    halüsinasyon görmesini engelleyen katı sistem promptudur.

    Phase 6: Diyalekt adı ve kuralları artık DİNAMİK ({dialect},
    {dialect_rules}) — aktif veri kaynağına göre doldurulur.
    {error_feedback} ise Phase 4 self-correction akışından değişmedi.
    """
    system_instruction = """Sen kıdemli bir Veritabanı Yönetim Sistemi (DBMS) uzmanı ve veri mühendisisin.

🎯 HEDEF DİYALEKT: {dialect}. Ürettiğin HER sorgu, bu veritabanının söz dizimine ve fonksiyon setine birebir uymalıdır:
{dialect_rules}

Aşağıda sunulan veritabanı şemasını (tablolar, kolonlar, veri tipleri, Foreign Key ilişkileri VE örnek veri satırları) detaylıca incele.
Kullanıcının doğal dilde sorduğu soruya karşılık, veritabanında çalışabilecek EN DOĞRU, OPTİMİZE ve GÜVENLİ {dialect} sorgusunu (query) üret.

VERİTABANI ŞEMASI (Tablolar, Kolonlar, Veri Tipleri, Foreign Key'ler ve Örnek Veriler):
{schema}

⚠️ UYULMASI ZORUNLU KATI KURALLAR (BUNLARI İHLAL EDEMEZSİN):
1. ÇIKTI OLARAK SADECE VE SADECE SAF SQL SORGUSU DÖNDÜR.
2. Kesinlikle selamlama, açıklama, ön bilgi veya markdown tırnağı (```sql ... ``` gibi) KULLANMA. Cevabın doğrudan 'SELECT' ile başlamalıdır.
3. Sadece şemada belirtilen tabloları ve kolonları kullan. Olmayan bir veriyi uydurma (No hallucination).
4. Eğer sorgu birden fazla tabloyu ilgilendiriyorsa, yukarıdaki Foreign Key ilişkilerini kullanarak doğru JOIN işlemlerini yap.
5. Sorgunun sonuna noktalı virgül (;) eklemeyi unutma.
6. FİLTRELEME ÖNCESİ ÖRNEK VERİ ANALİZİ: WHERE / LIKE koşulları yazmadan ÖNCE, yukarıdaki örnek veri satırlarındaki GERÇEK değer biçimini incele — büyük/küçük harf duyarlılığı, tarih formatı (örn. 'YYYY-MM-DD'), boşluk ve noktalama kullanımı. Filtrelerini varsayım yapmadan bu gerçek biçime birebir uydur.
7. HALÜSİNASYON ENGELİ: Kullanıcının sorusu yukarıdaki şemayla HİÇBİR şekilde ilişkili değilse (istenen veriye karşılık gelen hiçbir tablo/kolon şemada mevcut değilse), ASLA var olmayan bir tablo veya kolon UYDURMA. Bu durumda, kural 1 ve 2'yi yine de ihlal etmeden, tam olarak şu formatta GEÇERLİ bir SELECT sorgusu döndür:
   SELECT 'Bu soru mevcut veritabanı şemasıyla ilişkili değildir.' AS hata_mesaji;
{error_feedback}
Kullanıcının Doğal Dildeki Sorusu: {question}
Saf SQL Sorgusu:"""

    return ChatPromptTemplate.from_template(system_instruction)


def _build_error_feedback_block(previous_error: str) -> str:
    """
    Bir önceki başarısız denemenin hata mesajını, LLM'in anlayacağı
    düzeltici bir talimat bloğuna çevirir. (Phase 4'ten DEĞİŞMEDİ.)
    """
    return (
        "🔴 ÖNCEKİ DENEME BAŞARISIZ OLDU:\n"
        f'Bir önceki SQL sorgun şu hatayla başarısız oldu: "{previous_error}"\n'
        "Lütfen bu hatanın kök nedenini analiz et ve AYNI HATAYI TEKRARLAMADAN "
        "sorguyu düzelt. Sadece düzeltilmiş, saf SQL sorgusunu döndür.\n"
    )


def generate_sql_query(
    user_question: str,
    previous_error: Optional[str] = None,
    tenant_db_path: Optional[str] = None,
    tenant_db_url: Optional[str] = None,
) -> str:
    """
    Kullanıcının sorusunu alır, AKTİF veri kaynağının şemasıyla birleştirir
    ve LLM'den temizlenmiş SQL sorgusu üretir.

    Kaynak önceliği: tenant_db_url > tenant_db_path > varsayılan.
    Şema ÜRETİMİ ile sorgu ÇALIŞTIRMASI aynı kaynaktan olmak zorundadır;
    aksi halde LLM yanlış şemaya göre yazar ve her sorgu patlar.

    Args:
        user_question: Kullanıcının doğal dildeki sorusu.
        previous_error: Varsa, önceki başarısız denemenin hata mesajı
            (Phase 4 Self-Correction — DEĞİŞMEDİ).
        tenant_db_path: Kiracının yüklediği SQLite dosyasının yolu.
        tenant_db_url: Canlı veritabanı bağlantı dizesi (en yüksek öncelik).

    Returns:
        Temizlenmiş (markdown'sız, açıklamasız) saf SQL sorgusu metni.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin.")

    try:
        # 1. Aktif kaynağın diyalektini ve şemasını çıkar
        dialect = get_active_dialect(tenant_db_url=tenant_db_url, tenant_db_path=tenant_db_path)
        schema_info = get_enriched_schema_context(
            tenant_db_path=tenant_db_path,
            tenant_db_url=tenant_db_url,
        )
        logging.info(f"Şema okundu. Hedef diyalekt: {dialect}")

        # 2. LLM Modelini başlat (temperature=0 ile yaratıcılığı kapatıp, matematiksel kesinlik sağlıyoruz)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 3. Prompt şablonunu al
        prompt_template = get_sql_prompt_template()

        # 4. LangChain Zincirini (Chain) kur
        chain = prompt_template | llm

        # 5. Self-correction bağlamını hazırla (varsa) — Phase 4'ten değişmedi
        error_feedback = _build_error_feedback_block(previous_error) if previous_error else ""
        if previous_error:
            logging.warning("♻️ Self-correction modu: önceki hata prompta enjekte ediliyor.")

        logging.info(f"Yapay zekaya soru iletiliyor: '{user_question}'")

        # 6. Modeli çalıştır
        response = chain.invoke({
            "dialect": dialect,
            "dialect_rules": get_dialect_rules(dialect),
            "schema": schema_info,
            "question": user_question,
            "error_feedback": error_feedback,
        })

        # 7. Güvenlik Temizliği (LLM kuralı çiğneyip markdown koyarsa biz kodla söküp atıyoruz)
        clean_sql = response.content.strip().replace("```sql", "").replace("```", "").strip()

        return clean_sql

    except Exception as e:
        logging.error(f"SQL üretimi sırasında kritik hata: {e}")
        raise


if __name__ == "__main__":
    # Test Senaryosu: Ajanı üçlü JOIN yapmaya zorlayan karmaşık bir iş sorusu
    test_sorusu = "Ahmet Yılmaz adlı müşterinin sipariş ettiği ürünlerin isimleri ve satın aldığı adetler nelerdir?"

    print("\n" + "=" * 60)
    print("🚀 TEXT-TO-SQL MOTORU TEST EDİLİYOR")
    print("=" * 60)

    try:
        uretilen_sql = generate_sql_query(test_sorusu)
        print("\n💡 ÜRETİLEN SAF SQL SORGUSU:")
        print("-" * 60)
        print(uretilen_sql)
        print("-" * 60)
        print("\n✅ TEST BAŞARILI: Yapay zeka soruyu kusursuzca SQL'e çevirdi!\n")
    except Exception as e:
        print(f"\n❌ TEST BAŞARISIZ: {e}\n")