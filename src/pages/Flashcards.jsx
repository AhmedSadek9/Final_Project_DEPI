import { useState } from "react";

function Flashcards() {
  const [topic, setTopic] = useState("");
  const [cards, setCards] = useState([]);

  const generateCards = async () => {
    if (!topic.trim()) return;

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/flashcards",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            topic,
          }),
        }
      );

      const data = await response.json();

      const parsed = data.answer
        .split("===CARD===")
        .map((card) => {
          const question =
            card.match(/Q:\s*([\s\S]*?)(?=\nA:)/)?.[1]?.trim() || "";

          const answer =
            card.match(/A:\s*([\s\S]*)/)?.[1]?.trim() || "";

          return {
            question,
            answer,
          };
        })
        .filter(
          (card) =>
            card.question.length > 0 &&
            card.answer.length > 0
        );

      setCards(parsed);
    } catch (error) {
      console.error(error);
      alert("Failed to generate flashcards.");
    }
  };

  return (
    <div className="p-10 text-white">

      <h1 className="text-4xl font-bold text-slate-900 mb-8">
        🧠 Flashcards
      </h1>

      <input
        type="text"
        placeholder="Enter topic..."
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        className="w-full p-4 rounded-xl bg-slate-800 mb-6"
      />

      <button
        onClick={generateCards}
        className="bg-blue-600 px-5 py-3 rounded-xl mb-8"
      >
        Generate Flashcards
      </button>

      {cards.length > 0 && (
        <div className="space-y-6">

          {cards.map((card, index) => (
            <div
              key={index}
              className="bg-slate-800 p-6 rounded-xl shadow-lg"
            >

              <h2 className="text-2xl font-bold mb-4">
                Q: {card.question}
              </h2>

              <p className="text-lg leading-8 whitespace-pre-wrap">
                A: {card.answer}
              </p>

            </div>
          ))}

        </div>
      )}

    </div>
  );
}

export default Flashcards;