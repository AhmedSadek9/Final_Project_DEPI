import { useState, useEffect } from "react";

function StudyMode() {
  const [topic, setTopic] = useState("");
  const [result, setResult] = useState("");
  const [tutorAnswer, setTutorAnswer] = useState("");
  const [currentAgent, setCurrentAgent] = useState("");
  const [loading, setLoading] = useState(false);
  const [citations, setCitations] = useState([]);
  const [dots, setDots] = useState("");
  const [questions, setQuestions] = useState([]);
  const [writtenQuestions, setWrittenQuestions] = useState([]);
  const [writtenAnswers, setWrittenAnswers] = useState({});
  const [writtenFeedback, setWrittenFeedback] = useState({});
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState(0);
  const [explanations, setExplanations] = useState({});
  const [loadingExplanation, setLoadingExplanation] = useState({});
  const [tutorConversation, setTutorConversation] = useState([]);
  const [sendingTutor, setSendingTutor] = useState(false);
  const [interviewConversation, setInterviewConversation] = useState([]);
  const [interviewAnswer, setInterviewAnswer] = useState("");
  const [sendingInterview, setSendingInterview] = useState(false);

  useEffect(() => {
    if (!loading) {
      setDots("");
      return;
    }
    const interval = setInterval(() => {
      setDots((prev) => {
        if (prev === "...") return "";
        return prev + ".";
      });
    }, 400);
    return () => clearInterval(interval);
  }, [loading]);

  const callAgent = async (agentType) => {
    if (!topic.trim()) {
      alert("Please enter a topic.");
      return;
    }
    setLoading(true);
    setResult("");
    setCitations([]);
    setQuestions([]);
    setAnswers({});
    setSubmitted(false);
    setScore(0);

    setWrittenQuestions([]);
    setWrittenAnswers({});
    setWrittenFeedback({});

    setExplanations({});
    setLoadingExplanation({});
    setCurrentAgent(agentType);

    try {
      const response = await fetch(`http://127.0.0.1:8000/${agentType}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ topic }),
      });

      const data = await response.json();

      if (data.error) {
        alert(data.error);
        return;
      }

      if (agentType === "quiz") {

        setQuestions(data.questions || []);
        setWrittenQuestions(data.written_questions || []);
        setResult("");

      } else if (agentType === "tutor") {

        setTutorConversation([]);
        setResult(data.answer || "");

      } else if (agentType === "interview") {

        setInterviewConversation([]);
        setResult(data.answer || "");

      } else {

        setResult(data.answer || "");

      }

      setCitations(data.citations || []);
    } catch (error) {
      console.error(error);
      alert("Backend connection failed.");
    } finally {
      setLoading(false);
    }
  };

  const sendTutorMessage = async () => {
    if (!tutorAnswer.trim()) return;

    setSendingTutor(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/tutor-chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            topic,
            message: tutorAnswer,
            history: [
              ...tutorConversation,
              {
                student: tutorAnswer,
              },
            ],
          }),
        }
      );
      const data = await response.json();
      setTutorConversation((prev) => [
        ...prev,
        {
          student: tutorAnswer,
          tutor: data.answer,
        },
      ]);
      setTutorAnswer("");
    } catch (error) {
      console.error(error);
      alert("Failed to contact tutor.");
    } finally {
      setSendingTutor(false);
    }
  };

  const sendInterviewMessage = async () => {
    if (!interviewAnswer.trim()) return;

    setSendingInterview(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/interview-chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            topic,
            message: interviewAnswer,
            history: [
              ...interviewConversation,
              {
                student: interviewAnswer,
              },
            ],
          }),
        }
      );
      const data = await response.json();
      setInterviewConversation((prev) => [
        ...prev,
        {
          student: interviewAnswer,
          interviewer: data.answer,
        },
      ]);
      setInterviewAnswer("");
    } catch (error) {
      console.error(error);
      alert("Failed to contact tutor.");
    } finally {
      setSendingInterview(false);
    }
  };

  const selectAnswer = (questionIndex, option) => {
    setAnswers((prev) => ({
      ...prev,
      [questionIndex]: option,
    }));
  };

  const handleWrittenAnswer = (index, value) => {
    setWrittenAnswers((prev) => ({
      ...prev,
      [index]: value,
    }));
  };

  const submitQuiz = async () => {
    let correct = 0;

    questions.forEach((q, index) => {
      if (answers[index] === q.answer) {
        correct++;
      }
    });

    setScore(correct);
    setSubmitted(true);

    const feedback = {};

    for (let i = 0; i < writtenQuestions.length; i++) {

      const response = await fetch(
        "http://127.0.0.1:8000/evaluate-written",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: writtenQuestions[i],
            answer: writtenAnswers[i] || "",
          }),
        }
      );

      const data = await response.json();

      feedback[i] = data.feedback;
    }

    setWrittenFeedback(feedback);
  };

  const explainQuestion = async (question, index) => {
    if (explanations[index]) return;

    setLoadingExplanation((prev) => ({
      ...prev,
      [index]: true,
    }));

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/explanation",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: question.question,
            student_answer: answers[index] || "No Answer",
            correct_answer: question.answer,
          }),
        }
      );

      const data = await response.json();

      setExplanations((prev) => ({
        ...prev,
        [index]: data.answer,
      }));
    } catch (error) {
      console.error(error);
      alert("Failed to generate explanation.");
    }

    setLoadingExplanation((prev) => ({
      ...prev,
      [index]: false,
    }));
  };

  return (
    <div className="p-10 text-white">
      <h1 className="text-4xl font-bold text-slate-900 mb-8">📚 Study Mode</h1>

      <input
        type="text"
        placeholder="Enter a topic..."
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            callAgent("summary");
          }
        }}
        className="w-full p-4 rounded-xl bg-slate-800 mb-8"
      />

      <div className="flex flex-wrap gap-4 mb-8">
        <button
          onClick={() => callAgent("summary")}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-xl disabled:opacity-50"
        >
          Summary
        </button>
        <button
          onClick={() => callAgent("planner")}
          disabled={loading}
          className="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-xl disabled:opacity-50"
        >
          Learning Path
        </button>
        <button
          onClick={() => callAgent("quiz")}
          disabled={loading}
          className="bg-purple-600 hover:bg-purple-700 px-6 py-3 rounded-xl disabled:opacity-50"
        >
          Quiz
        </button>
        <button
          onClick={() => callAgent("tutor")}
          disabled={loading}
          className="bg-orange-600 hover:bg-orange-700 px-6 py-3 rounded-xl disabled:opacity-50"
        >
          Tutor
        </button>
        <button
          onClick={() => callAgent("interview")}
          disabled={loading}
          className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-xl disabled:opacity-50"
        >
          Interview
        </button>
      </div>

      <div className="bg-slate-800 rounded-xl p-6 min-h-[250px] whitespace-pre-wrap">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-16">
            <div className="w-14 h-14 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-5"></div>
            <p className="text-xl font-bold">AI is thinking{dots}</p>
            <p className="text-gray-400 mt-2">
              Please wait while your answer is generated.
            </p>
          </div>
        ) : questions.length > 0 ? (
          <div className="space-y-6">
            {questions.map((question, index) => (
              <div key={index} className="bg-slate-700 rounded-xl p-5">
                <h2 className="font-bold mb-4">Question {index + 1}</h2>
                <p className="mb-4">{question.question}</p>
                {question.options.map((option, optionIndex) => (
                  <label
                    key={optionIndex}
                    className={`block p-3 rounded-lg mb-2 cursor-pointer
                      ${
                        submitted
                          ? option === question.answer
                            ? "bg-green-600"
                            : answers[index] === option
                            ? "bg-red-600"
                            : "bg-slate-600"
                          : answers[index] === option
                          ? "bg-blue-600"
                          : "bg-slate-600"
                      }`}
                  >
                    <input
                      type="radio"
                      name={`q-${index}`}
                      checked={answers[index] === option}
                      disabled={submitted}
                      onChange={() => selectAnswer(index, option)}
                      className="mr-3"
                    />
                    {option}
                  </label>
                ))}

                {submitted && (
                  <div className="mt-4">
                    <button
                      onClick={() => explainQuestion(question, index)}
                      disabled={loadingExplanation[index]}
                      className="bg-indigo-600 hover:bg-indigo-700 px-4 py-2 rounded-lg text-sm disabled:opacity-50"
                    >
                      {loadingExplanation[index]
                        ? "Loading explanation..."
                        : explanations[index]
                        ? "Explanation loaded"
                        : "Explain this question"}
                    </button>

                    {explanations[index] && (
                      <div className="mt-3 bg-slate-600 rounded-lg p-4 text-sm text-gray-200 whitespace-pre-wrap">
                        {explanations[index]}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Written questions section */}
            {writtenQuestions.length > 0 && (
              <div className="mt-10">
                <h2 className="text-2xl font-bold mb-6">
                  Written Questions
                </h2>

                {writtenQuestions.map((question, index) => (
                  <div
                    key={index}
                    className="bg-slate-700 rounded-xl p-5 mb-6"
                  >
                    <h3 className="font-bold mb-3">
                      Question {index + 1}
                    </h3>

                    <p className="mb-4">
                      {question}
                    </p>

                    <textarea
                      rows={4}
                      value={writtenAnswers[index] || ""}
                      disabled={submitted}
                      onChange={(e) =>
                        handleWrittenAnswer(index, e.target.value)
                      }
                      className="w-full bg-slate-800 rounded-xl p-4"
                      placeholder="Write your answer..."
                    />

                    {submitted && writtenFeedback[index] && (
                      <div className="mt-4 bg-slate-800 rounded-xl p-4 whitespace-pre-wrap">
                        <h4 className="font-bold mb-2">
                          AI Feedback
                        </h4>
                        <p>{writtenFeedback[index]}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {!submitted && (
              <button
                onClick={submitQuiz}
                className="bg-green-600 px-8 py-3 rounded-xl font-bold"
              >
                Submit Quiz
              </button>
            )}

            {submitted && (
              <div className="bg-slate-700 p-6 rounded-xl">
                <h2 className="text-2xl font-bold">Score</h2>
                <p className="text-xl mt-2">
                  {score} / {questions.length}
                </p>
              </div>
            )}
          </div>
        ) : (
          result || "Your result will appear here."
        )}
      </div>

      {currentAgent === "tutor" && !loading && (
        <div className="mt-8">
          {tutorConversation.map((chat, index) => (
            <div
              key={index}
              className="bg-slate-700 rounded-xl p-4 mb-4 whitespace-pre-wrap"
            >
              <p>
                <strong>You:</strong> {chat.student}
              </p>
              <p className="mt-2">
                <strong>Tutor:</strong> {chat.tutor}
              </p>
            </div>
          ))}

          <input
            type="text"
            placeholder="Answer tutor question..."
            value={tutorAnswer}
            onChange={(e) => setTutorAnswer(e.target.value)}
            className="w-full p-4 rounded-xl bg-slate-800 mb-4"
          />
          <button
            onClick={sendTutorMessage}
            disabled={sendingTutor}
            className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-xl font-bold disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {sendingTutor ? "Tutor is thinking..." : "Send Answer"}
          </button>
        </div>
      )}

      {currentAgent === "interview" && !loading && (
        <div className="mt-8">
          {interviewConversation.map((chat, index) => (
            <div
              key={index}
              className="bg-slate-700 rounded-xl p-4 mb-4 whitespace-pre-wrap"
            >
              <p>
                <strong>You:</strong> {chat.student}
              </p>
              <p className="mt-2">
                <strong>Interviewer:</strong> {chat.interviewer}
              </p>
            </div>
          ))}

          <input
            type="text"
            placeholder="Answer interview question..."
            value={interviewAnswer}
            onChange={(e) => setInterviewAnswer(e.target.value)}
            className="w-full p-4 rounded-xl bg-slate-800 mb-4"
          />
          <button
            onClick={sendInterviewMessage}
            disabled={sendingInterview}
            className="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-xl font-bold disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {sendingInterview ? "Interviewer is thinking..." : "Send Answer"}
          </button>
        </div>
      )}

      {citations.length > 0 && (
        <div className="bg-slate-800 rounded-xl p-6 mt-8">
          <h2 className="text-2xl font-bold mb-5">📚 Sources Used</h2>
          {citations.map((citation, index) => (
            <div key={index} className="border-b border-slate-600 py-3">
              <p>
                <strong>Document:</strong> {citation.source}
              </p>
              <p>
                <strong>Page:</strong> {citation.page}
              </p>
              <p>
                <strong>Chapter:</strong> {citation.chapter}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default StudyMode;