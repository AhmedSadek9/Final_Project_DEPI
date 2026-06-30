import { useEffect, useState } from "react";

function RecentActivityCard() {

  const [history, setHistory] = useState([]);

  const loadHistory = async () => {

    try {

      const response = await fetch(
        "http://127.0.0.1:8000/history"
      );

      const data = await response.json();

      setHistory(data.history);

    } catch (err) {

      console.log(err);

    }

  };

  useEffect(() => {

    loadHistory();

  }, []);

  return (

    <div className="bg-white rounded-3xl shadow-lg p-6 h-full">

      <h2 className="text-xl font-bold mb-6">

        🕒 Recent Activity

      </h2>

      {

        history.length === 0 ?

        (

          <p className="text-slate-500">

            No activity yet.

          </p>

        )

        :

        (

          <div className="space-y-4">

            {

              history.map((item, index) => (

                <div

                  key={index}

                  className="border-b pb-3"

                >

                  <p className="font-semibold">

                    📁 {item.title}

                  </p>

                  <p className="text-sm text-slate-500">

                    {item.description}

                  </p>

                </div>

              ))

            }

          </div>

        )

      }

    </div>

  );

}

export default RecentActivityCard;