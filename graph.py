"""
graph.py

Orchestration layer (Separation of Concerns).

Bu modül YALNIZCA graf kablolamasını yapar: düğümler, kenarlar, döngü
kontrolü, derleme. İş mantığı başka yerdedir — araçlar agent.py'de, SQL
üretimi sql_generator.py'de, veri erişimi database.py'de, retrieval
rag_node.py'de.

PHASE 7 — STATİK ROUTER YERİNE TOOL CALLING DÖNGÜSÜ:

    ESKİ (Phase 3-6):
        [route_question] --'sql'--> generate_sql -> execute_sql -> [retry?]
                         --'rag'--> retrieve_policy -> END
        Tek yol seçilirdi; bileşik sorular yarım cevaplanırdı.

    YENİ (Phase 7):
                    ┌─────────────────────────┐
                    ▼                         │
        [ENTRY] -> agent --(tool_calls?)--> tools
                     │                        │
                     │ (araç çağrısı yok)     └── (sonuçlar mesajlara eklenir)
                     ▼
                   prune  ->  [END]

    - agent düğümü: araçlara bağlı LLM. Soruya göre SIFIR, BİR veya
      BİRDEN FAZLA aracı (aynı turda paralel) çağırabilir.
    - tools düğümü: ToolNode; çağrıları çalıştırıp ToolMessage üretir.
    - prune düğümü: tur bitiminde ARA araç trafiğini mesaj geçmişinden
      siler (checkpoint şişmesini/OOM riskini önler).

    DÖNGÜ SINIRI: MAX_TOOL_ITERATIONS. LLM aynı aracı sonsuza dek
    çağırmaya kalkarsa graf durur; kullanıcı asla sonsuz bekleyişte
    kalmaz ve maliyet patlamaz.

NOT — NİHAİ CEVAP BURADA ÜRETİLMEZ:
    agent düğümü bir "dispatcher"dır; görevi doğru araçları çağırmaktır.
    Kullanıcıya gidecek nihai, kaynağa bağlı (grounded) cevap
    chat_engine.py'deki Sözcü (Spokesperson) tarafından, YALNIZCA araç
    çıktılarına dayanılarak üretilir. Böylece Phase 6'da kurulan sert
    grounding ve deterministik sayı denetimi katmanı korunur.
"""

import logging
import os
from typing import Optional

from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent import TOOLS, AgentState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ======================================================================
# 1. SABİTLER
# ======================================================================

# Bir kullanıcı sorusu için en fazla kaç kez araç çağrısı turu yapılabilir.
# Sonsuz araç döngüsüne karşı sert emniyet valfi.
MAX_TOOL_ITERATIONS: int = 5

# Ajan (dispatcher) modeli. temperature=0: araç seçimi belirleyici olsun.
AGENT_MODEL: str = "gpt-4o-mini"


# ======================================================================
# 2. AJAN (DISPATCHER) SİSTEM PROMPT'U
# ======================================================================
# Bu prompt CEVAP YAZDIRMAZ; yalnızca doğru araçların çağrılmasını sağlar.
# Nihai cevabı chat_engine.py'deki Sözcü üretir.
_AGENT_SYSTEM_PROMPT = """Sen bir kurumsal veri asistanının yönlendirme katmanısın. Görevin, kullanıcının sorusunu cevaplamak için HANGİ VERİ KAYNAKLARINA başvurulması gerektiğine karar vermek ve ilgili araçları çağırmaktır.

ARAÇLARIN:
- query_database_tool: yapılandırılmış veri (müşteriler, siparişler, ürünler, stok, fiyat, adet, satış rakamları, tarihler).
- search_documents_tool: yazılı politika ve doküman bilgisi (iade, kargo, garanti, ödeme koşulları, prosedürler, sözleşmeler).

KURALLAR:
1. Soru HER İKİ tür bilgiyi de istiyorsa (örneğin "geçen ayki satışları VE iade politikasını göster"), İKİ ARACI DA çağır. Tek araçla yetinme.
2. Araçlara gönderdiğin `question` parametresi TEK BAŞINA anlaşılır olmalıdır. Kullanıcı "peki ya onlar için?" gibi eksiltili bir soru sorduysa, konuşma geçmişine bakarak soruyu tamamla (örneğin "VIP müşteriler için iade süresi").
3. Bir araç hata döndürürse AYNI aracı aynı parametrelerle tekrar çağırma; sistem kendi içinde zaten yeniden deniyor.
4. Soru hiçbir veri kaynağı gerektirmiyorsa (selamlaşma, teşekkür, sohbet) hiçbir aracı çağırma.
5. Kendi bilginle cevap YAZMA. Senin işin veri toplamaktır, cevaplamak değil. Araç çağırmayı bitirdiğinde kısa bir onay metni yeterlidir."""


# ======================================================================
# 3. DÜĞÜMLER
# ======================================================================
def _count_tool_iterations(messages: list) -> int:
    """Bu turda kaç kez araç çağrısı yapıldığını sayar."""
    return sum(
        1
        for m in messages
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )


def _build_agent_node(model):
    """Ajan (dispatcher) düğümünü üretir."""

    def agent_node(state: AgentState) -> dict:
        logging.info("🧠 Düğüm: 'agent' (araç seçimi) çalışıyor...")
        messages = state["messages"]

        # Emniyet valfi: araç döngüsü sınırı aşıldıysa daha fazla araç
        # çağrısına izin verme; boş içerikli bir AIMessage döndürerek
        # döngüyü sonlandır (route fonksiyonu bunu END'e yönlendirir).
        if _count_tool_iterations(messages) >= MAX_TOOL_ITERATIONS:
            logging.warning(
                f"🛑 Araç döngüsü sınırı ({MAX_TOOL_ITERATIONS}) aşıldı; toplama durduruldu."
            )
            return {"messages": [AIMessage(content="Veri toplama tamamlandı.")]}

        response = model.invoke([SystemMessage(content=_AGENT_SYSTEM_PROMPT), *messages])
        return {"messages": [response]}

    return agent_node


def prune_node(state: AgentState) -> dict:
    """
    Tur bitiminde ARA araç trafiğini mesaj geçmişinden siler.

    Neden: add_messages her şeyi biriktirir. Araç çıktıları (SQL satırları,
    doküman parçaları) büyük olabilir ve her tur checkpoint'e yazılır —
    Phase 5.2'de bilinçle çözülen "state şişmesi/OOM" sorununun aynısı
    geri dönerdi. Ayrıca her yeni turda tüm eski araç dökümleri LLM'e
    tekrar gönderilir, token maliyeti şişerdi.

    Ne korunur: kullanıcı soruları (HumanMessage) ve araç çağrısı
    İÇERMEYEN asistan mesajları — yani konuşmanın anlamlı akışı. Böylece
    "peki ya VIP müşteriler?" gibi eksiltili takip soruları hâlâ bağlam
    bulur.

    Ne silinir: tool_calls içeren AIMessage'lar ve ToolMessage'lar.
    İkisi BİRLİKTE silinir; yalnızca birini silmek, sağlayıcının
    "tool_call'a karşılık gelen sonuç yok" hatası vermesine yol açardı.
    """
    removals = [
        RemoveMessage(id=m.id)
        for m in state["messages"]
        if m.id is not None
        and (isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and getattr(m, "tool_calls", None)))
    ]
    if removals:
        logging.info(f"🧹 Prune: {len(removals)} ara araç mesajı geçmişten temizlendi.")
    return {"messages": removals}


def route_after_agent(state: AgentState) -> str:
    """
    Ajanın son mesajına bakar: araç çağrısı var mı?

    Saf yönlendirme fonksiyonu — yan etkisi yoktur, state'i değiştirmez.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "prune"


# ======================================================================
# 4. GRAF KURULUMU
# ======================================================================
def build_graph(model=None) -> CompiledStateGraph:
    """
    Tool-calling ajanı grafiğini kurar ve derler.

    Topoloji:
        [ENTRY] -> agent --(tool_calls var)--> tools --+
                     |                                  |
                     |  (araç çağrısı yok)              +--> agent (döngü)
                     v
                   prune -> [END]

    Args:
        model: Test enjeksiyonu için opsiyonel chat modeli. Verilmezse
            araçlara bağlanmış ChatOpenAI kullanılır.

    Returns:
        CompiledStateGraph: MemorySaver checkpointer'lı, invoke/stream
        arayüzünü sunan derlenmiş graf.
    """
    if model is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY bulunamadı! Ajan modeli başlatılamıyor. "
                "Lütfen .env dosyanızı kontrol edin."
            )
        model = ChatOpenAI(model=AGENT_MODEL, temperature=0).bind_tools(TOOLS)

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", _build_agent_node(model))
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("prune", prune_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "prune": "prune"},
    )
    # Araçlar çalıştıktan sonra karar yine ajana döner: yeterli veri
    # toplandı mı, yoksa başka bir araca daha mı ihtiyaç var?
    workflow.add_edge("tools", "agent")
    workflow.add_edge("prune", END)

    # Konuşma hafızası: thread_id bazlı in-memory checkpoint.
    memory = MemorySaver()

    logging.info(
        "📐 Graph derlendi (Tool Calling): agent <-> tools döngüsü "
        f"(maks. {MAX_TOOL_ITERATIONS} tur) -> prune -> END"
    )

    return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    # Sağlamlık testi (gerçek OpenAI çağrıları yapar — OPENAI_API_KEY gerekir).
    from langchain_core.messages import HumanMessage

    test_questions = [
        "En pahalı 3 ürün hangisi?",  # -> query_database_tool
        "İade politikanız nedir?",  # -> search_documents_tool
        "En pahalı ürünü ve iade politikasını birlikte söyler misin?",  # -> HER İKİSİ
    ]

    app = build_graph()

    for i, test_question in enumerate(test_questions, start=1):
        print("\n" + "=" * 60)
        print(f"🚀 SORU {i}: {test_question}")
        print("=" * 60)

        config = {"configurable": {"thread_id": f"graph-selftest-{i}"}}
        final_state = app.invoke(
            {
                "messages": [HumanMessage(content=test_question)],
                "tenant_db_path": None,
                "tenant_db_url": None,
                "tenant_vector_store_path": None,
            },
            config,
        )

        print("\n📊 MESAJ AKIŞI:")
        print("-" * 60)
        for m in final_state["messages"]:
            kind = type(m).__name__
            preview = (m.content or "")[:200].replace("\n", " ")
            print(f"  {kind}: {preview}")
        print("-" * 60)