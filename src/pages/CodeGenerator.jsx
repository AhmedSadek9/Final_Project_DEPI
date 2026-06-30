import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

function CodeGenerator() {
  const [topic, setTopic] = useState("");
  const [language, setLanguage] = useState("Python");
  const [code, setCode] = useState("");
  const [explanation, setExplanation] = useState("");
  const [timeComplexity, setTimeComplexity] = useState("");
  const [spaceComplexity, setSpaceComplexity] = useState("");
  const [bestPractices, setBestPractices] = useState("");
  const [commonMistakes, setCommonMistakes] = useState("");
  const [loading, setLoading] = useState(false);

  const generateCode = async () => {
    if (!topic.trim()) return;
    setLoading(true);

    setCode("");
    setExplanation("");
    setTimeComplexity("");
    setSpaceComplexity("");
    setBestPractices("");
    setCommonMistakes("");
    try {
      const response = await fetch(
        "http://127.0.0.1:8000/code-generator",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            topic,
            language,
          }),
        }
      );
      const data = await response.json();
      console.log(data);
      setCode(data.code || "");
      setExplanation(data.explanation || "");
      setTimeComplexity(data.time_complexity || "");
      setSpaceComplexity(data.space_complexity || "");
      setBestPractices(data.best_practices || "");
      setCommonMistakes(data.common_mistakes || "");
    } catch (error) {
      console.error(error);
      alert("Failed to generate code.");
    } finally {
      setLoading(false);
    }
  };

  const downloadCode = () => {
    if (!code) return;

    const extensions = {
      Python: "py",
      Java: "java",
      "C++": "cpp",
      JavaScript: "js",
      "C#": "cs",
    };

    const extension = extensions[language] || "txt";

    const blob = new Blob([code], {
      type: "text/plain",
    });

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");

    link.href = url;

    link.download = `generated_code.${extension}`;

    link.click();

    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-10">
      <h1 className="text-4xl font-bold text-slate-900 mb-8">
        💻 Code Generator
      </h1>

      <input
        type="text"
        placeholder="Enter algorithm, data structure, or programming problem..."
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        className="w-full p-4 rounded-xl bg-slate-800 text-white mb-6"
      />

      <select
        value={language}
        onChange={(e) => setLanguage(e.target.value)}
        className="w-full p-4 rounded-xl bg-slate-800 text-white mb-6"
      >
        <option>Python</option>
        <option>C++</option>
        <option>Java</option>
        <option>JavaScript</option>
        <option>C#</option>
      </select>

      <button
        onClick={generateCode}
        disabled={loading}
        className="bg-blue-600 hover:bg-blue-700 px-8 py-3 rounded-xl font-bold disabled:opacity-50"
      >
        {loading ? "🤖 Generating Code..." : "Generate Code"}
      </button>

      {code && (
        <>
          <div className="bg-slate-800 rounded-xl p-6 mt-8">
            <h2 className="text-2xl font-bold mb-4 text-white">💻 Code</h2>

            <div className="flex gap-4 mb-4">
              <button
                onClick={() => navigator.clipboard.writeText(code)}
                className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-white"
              >
                📋 Copy Code
              </button>

              <button
                onClick={downloadCode}
                className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg text-white"
              >
                💾 Download Code
              </button>
            </div>

            <SyntaxHighlighter
              language={language.toLowerCase()}
              style={oneDark}
              showLineNumbers
            >
              {code}
            </SyntaxHighlighter>
          </div>

          <div className="bg-slate-800 rounded-xl p-6 mt-6 text-white">
            <h2 className="text-2xl font-bold mb-4">📖 Explanation</h2>
            <p>{explanation}</p>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mt-6">
            <div className="bg-slate-800 rounded-xl p-6 text-white">
              <h2 className="text-xl font-bold mb-3">⏱ Time Complexity</h2>
              <p>{timeComplexity}</p>
            </div>

            <div className="bg-slate-800 rounded-xl p-6 text-white">
              <h2 className="text-xl font-bold mb-3">💾 Space Complexity</h2>
              <p>{spaceComplexity}</p>
            </div>
          </div>

          <div className="bg-slate-800 rounded-xl p-6 mt-6 text-white">
            <h2 className="text-2xl font-bold mb-4">✅ Best Practices</h2>
            <p>{bestPractices}</p>
          </div>

          <div className="bg-slate-800 rounded-xl p-6 mt-6 text-white">
            <h2 className="text-2xl font-bold mb-4">⚠ Common Mistakes</h2>
            <p>{commonMistakes}</p>
          </div>
        </>
      )}
    </div>
  );
}

export default CodeGenerator;