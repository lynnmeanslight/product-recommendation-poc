import React, { useState } from "react";
import logo from "./assets/airgas_logo.svg"; // Assuming your logo is in the src folder and named logo.svg

function App() {
  const api_url = "http://54.252.245.80/api";
  const [userId, setUserId] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [pastPurchases, setPastPurchases] = useState([]);
  const [retrievingTime, setRetrievingTime] = useState(null);

  const convertSecondsToMilliseconds = (seconds) => {
    // Multiply seconds by 1000 to convert to milliseconds
    const milliseconds = seconds * 1000;
    // Round the milliseconds to 2 decimal places and convert to a number (optional)
    return parseFloat(milliseconds.toFixed(2));
  };

  const fetchRecommendations = async () => {
    const response = await fetch(`${api_url}/users/${userId}`);
    const data = await response.json();
    setRecommendations(data.body);
    setRetrievingTime(data.retrieving_time);
  };

  const fetchPastPurchases = async () => {
    const response = await fetch(`${api_url}/users/${userId}/past_purchases`);
    const data = await response.json();
    setPastPurchases(data);
  };

  const fetchData = async () => {
    fetchRecommendations();
    fetchPastPurchases();
  };

  return (
    <div style={{ fontFamily: "'Roboto', 'sans-serif'" }}>
      <nav
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "1rem",
          backgroundColor: "#f0f0f0",
        }}
      >
        <img
          src={logo}
          alt="Logo"
          style={{ height: "50px", marginLeft: "100px" }}
        />
        <p style={{ color: "#006072", fontWeight: "bold" }}>
          Product Recommendation System
        </p>
      </nav>
      {/* Your existing component code here */}
      <div style={{ margin: "50px" }}>
        <p>There are 206,209 users in the sample data.</p>
        <p>You can type ID from 1 to 206209.</p>

        <p
          style={{ color: "#006272", fontWeight: "bold", marginBottom: "0px" }}
        >
          Enter User ID:
        </p>
        <input
          style={{
            padding: "10px",
            borderRadius: "5px",
            marginTop: "0px",
            border: "1px solid #ccc",
          }}
          placeholder="Enter User ID"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
        />
        <button
          onClick={fetchData}
          style={{
            backgroundColor: "#006072", // Green background
            color: "white", // White text
            padding: "10px 20px", // Padding around the text
            margin: "10px", // Margin around the button
            border: "none", // No border
            borderRadius: "5px", // Rounded corners
            cursor: "pointer", // Pointer cursor on hover
            fontSize: "16px", // Larger text
          }}
        >
          Fetch Data
        </button>
        <br />
        {retrievingTime
          ? `Time needed for retrieving recommended products using Pinecone: ${convertSecondsToMilliseconds(
              retrievingTime
            )} milliseconds.`
          : ""}
        {/* Recommendations Table */}

        <div style={{ display: "flex" }}>
          {recommendations.length !== 0 ? (
            <div>
              <p style={{ fontWeight: "bold" }}>
                Recommendations for User ID: {userId}
              </p>
              <table
                style={{
                  width: "500px",
                  backgroundColor: "#",
                  borderCollapse: "collapse",
                  textAlign: "center",
                  border: "2px solid #006072",
                }}
              >
                <thead>
                  <tr>
                    <th
                      style={{
                        width: "25%",
                        borderBottom: "1px solid #379777",
                        padding: "8px",
                        textAlign:"left",
                        backgroundColor: "#006072", // Add background color to header
                        color: "white", // Change text color to white for header
                        borderRight: "1px solid #379777",
                      }}
                    >
                      Product ID
                    </th>
                    <th
                      style={{
                        width: "50%",
                        borderBottom: "1px solid #379777",
                        padding: "8px",
                        backgroundColor: "#006072", // Add background color to header
                        color: "white", // Change text color to white for header
                        borderRight: "1px solid #379777",
                      }}
                    >
                      Product Name
                    </th>
                    <th
                      style={{
                        width: "25%",
                        borderBottom: "1px solid #379777",
                        padding: "8px",
                        backgroundColor: "#006072", // Add background color to header
                        color: "white", // Change text color to white for header
                        borderRight: "1px solid #379777",
                        textAlign:"right",
                      }}
                    >
                      Score
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {recommendations.map((item, index) => (
                    <tr key={index}>
                      <td
                        style={{
                          borderBottom: "1px solid #379777",
                          padding: "8px",
                          textAlign:"left",
                          borderRight: "1px solid #379777",
                        }}
                      >
                        {item.id}
                      </td>
                      <td
                        style={{
                          borderBottom: "1px solid #379777",
                          padding: "8px",
                          borderLeft: "1px solid #379777",
                          textAlign: "left",
                        }}
                      >
                        {item.name}
                      </td>
                      <td
                        style={{
                          borderBottom: "1px solid #379777",
                          padding: "8px",
                          borderLeft: "1px solid #379777",
                          textAlign:"right",
                        }}
                      >
                        {item.scores}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            "No recommendations found for current user id"
          )}

          {pastPurchases.length !== 0 ? (
            <div style={{ marginLeft: "50px" }}>
              <p style={{ fontWeight: "bold" }}>
                Past Purchases by User ID: {userId}
              </p>
              <table
                style={{
                  width: "500px",
                  borderCollapse: "collapse",
                  textAlign: "center",
                  border: "2px solid #006072",
                }}
              >
                <thead>
                  <tr>
                    <th
                      style={{
                        width: "25%",
                        borderBottom: "1px solid #379777",
                        padding: "8px",
                        textAlign:"left",
                        backgroundColor: "#006072", // Add background color to header
                        color: "white", // Change text color to white for header
                        borderRight: "1px solid #379777",
                      }}
                    >
                      Product ID
                    </th>
                    <th
                      style={{
                        width: "50%",
                        borderBottom: "1px solid #379777",
                        padding: "8px",
                        backgroundColor: "#006072", // Add background color to header
                        color: "white", // Change text color to white for header
                        borderRight: "1px solid #379777",
                      }}
                    >
                      Product Name
                    </th>
                    <th
                      style={{
                        width: "25%",
                        borderBottom: "1px solid #379777",
                        padding: "8px",
                        backgroundColor: "#006072", // Add background color to header
                        color: "white", // Change text color to white for header
                        borderRight: "1px solid #379777",
                      }}
                    >
                      Quantity
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {pastPurchases.map((item, index) => (
                    <tr key={index}>
                      <td
                        style={{
                          borderBottom: "1px solid #379777",
                          padding: "8px",
                          borderRight: "1px solid #379777",
                          textAlign:"left",
                        }}
                      >
                        {item.product_id}
                      </td>
                      <td
                        style={{
                          borderBottom: "1px solid #379777",
                          padding: "8px",
                          textAlign: "left",
                          borderRight: "1px solid #379777",
                        }}
                      >
                        {item.product_name}
                      </td>
                      <td
                        style={{
                          borderBottom: "1px solid #379777",
                          padding: "8px",
                          borderLeft: "1px solid #379777",
                        }}
                      >
                        {item.total_orders}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            "No past purchases found for current user id"
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
