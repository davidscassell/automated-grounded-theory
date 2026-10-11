"""Miniature interviews. Synthetic, so tests do not need a live model or a key."""

CORPUS = [
    {
        "id": "alex",
        "name": "Alex",
        "text": "I spent the morning packing my suitcase. The room looked empty once the books were gone.",
    },
    {
        "id": "blair",
        "name": "Blair",
        "text": "Packing bags up with boxes took all night. I kept finding letters I could not throw away.",
    },
    {
        "id": "casey",
        "name": "Casey",
        "text": "I was crying at the station when the train doors closed. My sister waved until she disappeared.",
    },
    {
        "id": "drew",
        "name": "Drew",
        "text": "They hired me on a Friday. Starting the new job meant moving before the month ended.",
    },
]

CONTEXT = {
    "research_question": "How do students describe the days around leaving home?",
    "population": "Undergraduates in their first term",
    "sensitizing_concepts": ["leaving home"],
}
