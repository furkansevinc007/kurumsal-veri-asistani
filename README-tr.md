🇹🇷 Türkçe (buradasınız) | [🇬🇧 Read in English (README.md)](README.md)

# 🤖 Kurumsal Veri Asistanı

> Bir sohbet botu değil, **Otonom Karar Destek Sistemi**. Sanal bir veri analisti gibi çalışır: doğal dilde sorulan bir soruyu anlar, hangi veri kaynaklarına başvuracağına **kendisi karar verir**, gerektiğinde kendi SQL sorgusunu yazıp veritabanında çalıştırır, kurumsal PDF belgelerinizden ilgili bölümleri bulur ve tüm bunları tek, tutarlı ve kaynağa dayalı bir cevapta birleştirir.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11%20|%203.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.60-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangGraph-1.2-1C3C3C?logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/LangChain-Core%201.4-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/ChromaDB-1.5-FF6F61" alt="ChromaDB">
  <img src="https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?logo=sqlalchemy&logoColor=white" alt="SQLAlchemy">
  <img src="https://img.shields.io/badge/Lisans-MIT-green" alt="Lisans">
</p>

---

## 📸 Demo ve Ekran Görüntüleri

|            Karşılama Ekranı            |    Bileşik Soru + Kaynaklar    |             Geliştirici Görünümü              |
|:--------------------------------------:|:------------------------------:|:---------------------------------------------:|
| ![Karşılama](docs/ekran_karsilama.png) | ![Cevap](docs/ekran_cevap.png) | ![Geliştirici Görünümü](docs/ekran_debug.png) |

---

## 🚀 Bu Neden Sıradan Bir Chatbot Değil?

Sıradan bir yapay zekâ asistanı yalnızca yüklü metinler arasından cevap arar. Bu sistem ise bir **ajan** (otonom karar veren bir yazılım) olarak tasarlandı. Her soru için, o an **hangi araçları kullanacağına kendisi karar verir** ve gerektiğinde **birden fazla kaynağı aynı anda** kullanabilir.

- **🧠 Kendi Kararını Veren Yapay Zekâ.** Asistanın iki temel yeteneği vardır: veritabanında sorgulama yapmak ve belgelerde arama yapmak. Sorunun türüne göre hangisini (ya da her ikisini birden) kullanacağına kendisi karar verir. Örneğin _"Geçen ayın en çok satan ürününü **ve** VIP iade politikamızı göster"_ dediğinizde, aynı anda hem veritabanını sorgular hem de belgeleri tarar; ardından iki bilgiyi tek bir bütünlüklü cevapta birleştirir.

- **🛠️ Hatasını Kendi Düzelten SQL Yazarı.** Asistan, sorunuzu bir veritabanı sorgusuna çevirir ve çalıştırır. Eğer sorgu ilk denemede hata verirse, **hatayı okur, nedenini analiz eder ve sorguyu kendisi düzelterek yeniden dener** — kullanıcıya bir hata döndürmeden önce **3 kez**. Yani "çalışmadı" demeden önce sorunu kendi çözmeye çalışır.

- **🌐 Farklı Veritabanlarıyla Uyumlu.** Bağlandığı veritabanının türünü tanır ve sorgularını ona göre yazar. **SQLite, PostgreSQL, MySQL, MariaDB, MSSQL ve Oracle** desteklenir.

- **🔒 Uydurmaya Karşı Katı Koruma (Grounding).** Yapay zekânın en bilinen riski "bilgi uydurmasıdır". Bu sistem, asistanın **yalnızca gerçek verilerden** cevap vermesini zorunlu kılar. Üstüne, ek bir güvenlik katmanı olarak: cevaptaki **her sayıyı otomatik olarak kontrol eder** ve kaynak veride bulunmayan bir rakam varsa kullanıcıyı görünür bir uyarıyla bilgilendirir. Örneğin belgede "45 gün" yazarken asistanın kafasından "30 gün" demesini yakalamak için tasarlanmıştır.

- **📎 Şeffaf ve Güvenilir Kaynak Gösterimi.** Her cevabın sonunda, bilginin hangi belgeden (dosya adı ve sayfa numarasıyla) veya hangi veritabanından geldiği **otomatik olarak** listelenir. Bu liste yapay zekâ tarafından uydurulmaz; gerçek kaynaklardan derlenir.

---

## ✨ Temel Özellikler

### Hibrit Veri Stratejisi (Akıllı Yönlendirme)
Asistan üç farklı kaynak türüyle çalışabilir ve birden fazlası bağlıysa şu öncelik sırasını izler:

```
Canlı Veritabanı Bağlantısı  >  Yüklenen SQLite (.db) dosyası  >  Örnek demo verisi
```

Belgeler ayrı bir yetenekle ele alındığından, bir soru aynı anda hem sayısal verilere hem de yazılı politika metinlerine ulaşabilir.

### Çok Kullanıcılı İzolasyon ve Güvenlik (SaaS Mimarisi)
- **Her oturuma özel kimlik.** Her kullanıcı oturumu benzersiz bir kimlik (UUID) alır; yüklenen tüm dosyalar ve veriler o oturuma ayrılmış, **izole** bir alanda tutulur.
- **Belgelerin birbirine karışması matematiksel olarak imkânsız.** Her yükleme, **yepyeni ve benzersiz** bir alana yazılır. Eski bir verinin yeni bir aramaya karışması mümkün değildir.
- **Şifreler yapay zekâya asla gösterilmez.** Canlı veritabanı bağlantı bilgileri, yapay zekânın **hiçbir zaman göremeyeceği** güvenli bir kanaldan iletilir. Bağlantı bilgileri kayıtlarda ve ekranda daima maskelenir.
- **Tasarımı gereği "salt okunur".** Asistan yalnızca **veri okuyabilir** (`SELECT`). Veri silme, değiştirme veya ekleme (`DROP` / `DELETE` / `UPDATE`) komutları çalıştırılmadan önce reddedilir. Verileriniz asla değiştirilemez.

### Akıllı Otomatik Temizlik (Geçici Veri Yönetimi)
- Sistem her açıldığında, **12 saatten eski** oturum verileri otomatik olarak silinir. Böylece sunucu beklenmedik şekilde kapansa bile veriler birikmez.
- Aktif oturumlar korunur; kullanan bir kullanıcının verisi asla erken silinmez.
- Bir kaynağı kaldırdığınızda, verisi diskten anında temizlenir.

### Şeffaf "Geliştirici Görünümü"
Tek dokunuşla açılan bu panel, her cevap için şunları gösterir: asistanın **hangi araçları kullandığı**, çalıştırdığı **tam SQL sorgusu**, elde ettiği **ham veriler** ve **kaynak listesi**. Yapay zekânın nasıl düşündüğüne tam şeffaflık sağlar — teknik ekipler için ideal bir denetim aracı.

### Profesyonel Kullanıcı Arayüzü
- **Akıllı karşılama ekranı:** Henüz veri bağlanmamışken kullanıcıyı adım adım yönlendirir; bir kaynak bağlanınca kendiliğinden kaybolur.
- **Anlık bağlantı testi:** Girilen veritabanı adresi, kullanıcı ilk sorusunu sormadan **bağlanma anında** doğrulanır.
- **Sağlam dosya yönetimi:** Çoklu PDF yükleme, dosya türü doğrulama ve hata durumunda kullanıcıyı asla "kilitlemeyen" akış.
- **Kelime kelime akan cevaplar** ve işlem sırasında canlı durum bildirimleri.

---

## 🏗️ Sistem Mimarisi

### Araç Kullanım Döngüsü (Tool-Calling)
Motor, konuşma hafızasına sahip bir LangGraph iş akışı üzerine kuruludur.

```
                     ┌─────────────────────────────┐
                     ▼                             │
   [Kullanıcı Sorusu] → ajan ──(araç gerekli mi?)──→ araçlar
                         │                          │
                         │ (araç gerekmez)          └─ sonuçlar hafızaya eklenir
                         ▼
                     temizlik ──→ [BİTİŞ] ──→ Sözcü (kaynağa dayalı cevap)
```

1. **Ajan (Yönlendirici):** `gpt-4o-mini` modeli. Hangi araçların (sıfır, bir veya aynı anda birkaç) çağrılacağına karar verir. Nihai cevabı **kendisi yazmaz**.
2. **Araçlar:** Veritabanı sorgusu (kendi kendini düzelten SQL akışıyla) veya belge araması burada çalışır. Kullanıcıya özel bilgiler (dosya yolları, şifreler) araçlara güvenli kanaldan iletilir.
3. **Temizlik:** Her turdan sonra, büyük ara veriler (SQL satırları, belge parçaları) hafızadan silinir; böylece uzun konuşmalarda sistem yavaşlamaz veya şişmez.
4. **Sözcü (Servis Katmanı):** Ayrı bir model, nihai Türkçe cevabı **yalnızca araç sonuçlarından**, katı uydurma-önleme kurallarıyla üretir. Cevabı üretme işini karar verme işinden ayırmak, uydurmaya karşı korumanın güvence altına alınmasını sağlayan tasarım tercihidir.

Sonsuz döngüleri önlemek için bir güvenlik sınırı (`MAX_TOOL_ITERATIONS`) bulunur.

### Belge Arama Altyapısı (Yükleme Anında İşleme)
PDF'ler **yalnızca bir kez**, yükleme anında işlenir ve o oturuma özel kalıcı bir arama deposuna kaydedilir. Sonraki sorularda belge yeniden işlenmez — bu sayede tekrar eden sorular ek maliyet oluşturmaz ve anında yanıtlanır. Her belge parçası, kaynak dosya adı ve sayfa numarasıyla birlikte saklanır.

### Üç Katmanlı Mimari (Separation of Concerns)

| Katman | Dosyalar | Sorumluluk |
| :--- | :--- | :--- |
| **Arayüz** | `app.py` | Sunum, oturum yönetimi, dosya yükleme, karşılama ekranı. |
| **Servis** | `chat_engine.py` | Motor olaylarını arayüze çevirir; kaynağa dayalı Sözcü'yü ve doğrulamayı çalıştırır. |
| **Motor** | `graph.py`, `agent.py`, `database.py`, `sql_generator.py`, `rag_node.py` | Araç döngüsü, araçlar, veritabanı erişimi, SQL üretimi, belge yönetimi. |

---

## ⚙️ Kurulum ve Çalıştırma

### Gereksinimler
- Python **3.11** veya **3.12**
- Bir **OpenAI API anahtarı**

### 1. Projeyi klonlayın
```bash
git clone https://github.com/furkansevinc007/kurumsal-veri-asistani.git
cd kurumsal-veri-asistani
```

### 2. Sanal ortam oluşturun
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### 4. OpenAI API anahtarınızı tanımlayın

**Yerel geliştirme** — proje kök dizininde bir `.env` dosyası oluşturun:
```env
OPENAI_API_KEY=sk-anahtariniz-buraya
```

**Streamlit Community Cloud** — bunun yerine **Settings → Secrets** bölümüne ekleyin:
```toml
OPENAI_API_KEY = "sk-anahtariniz-buraya"
```

### 5. Çalıştırın
```bash
streamlit run app.py
```
Sol üstteki **›** simgesinden kenar menüsünü açın; bir `.db` / PDF yükleyin veya canlı bir veritabanına bağlanın, ardından sorularınızı sormaya başlayın.

> **☁️ Streamlit Community Cloud'a mı yükleyeceksiniz?** `app.py` dosyasının en üstünde kritik bir SQLite uyum yaması bulunur (sistemin `sqlite3` modülünü `pysqlite3-binary` ile değiştirir), çünkü Cloud'un varsayılan SQLite sürümü ChromaDB için fazla eskidir. Bu sorun zaten çözülmüştür — yalnızca `pysqlite3-binary` paketini `requirements.txt` içinde tutun.

---

## 🧰 Kullanılan Teknolojiler

| Kategori | Teknoloji |
| :--- | :--- |
| **Dil** | Python 3.11 / 3.12 |
| **Arayüz** | Streamlit |
| **Ajan Orkestrasyonu** | LangGraph (araç çağıran iş akışı, `ToolNode`, `InjectedState`, `MemorySaver`) |
| **Yapay Zekâ Çatısı** | LangChain |
| **Modeller** | OpenAI `gpt-4o-mini` (akıl yürütme + cevap), `text-embedding-3-small` (belge işleme) |
| **Vektör Deposu** | ChromaDB (kalıcı, oturuma özel) |
| **Veritabanı / ORM** | SQLAlchemy 2.0 — SQLite, PostgreSQL, MySQL, MariaDB, MSSQL, Oracle |
| **Belge İşleme** | pypdf |

---

## ⚠️ Notlar ve Sınırlamalar
- Sistem yalnızca **veri okuma** işlemi yapar. Canlı bağlantılar için salt okunur bir veritabanı kullanıcısı önerilir.
- Streamlit Community Cloud'da depolama **geçicidir** — yüklenen veriler uygulama yeniden başladığında kalıcı olmaz. 12 saatlik otomatik temizlik ve "yeniden yükleyin" akışları bu durumu sorunsuz yönetir.
- MSSQL ve Oracle, SQL üretici tarafından desteklenir; ancak sürücüleri sistem düzeyinde ek bileşenler gerektirdiğinden Streamlit Cloud'da çalışmaz, kendi sunucunuzu kullanmanız gerekir.

---

## 📄 Lisans
MIT Lisansı ile dağıtılmaktadır. Ayrıntılar için `LICENSE` dosyasına bakınız.

---

<p align="center"><i>Üretim odaklı Agentic AI mimarisinin bir gösterimi olarak geliştirildi — otonom yönlendirme, kendi kendini düzeltme, çok kullanıcılı izolasyon ve güvenilir kaynak bağlılığı.</i></p>
