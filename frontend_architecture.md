# Frontend Architecture Plan: Smart-Inventory

## 1. Dashboard UI/UX Layout

The Smart-Inventory dashboard is tailored directly to the Supermarket Manager, emphasizing actionable insights alongside transparent predictions.

*   **Global Navigation (Sidebar):** Minimalist left sidebar containing key links: Dashboard, Run Forecast (Custom Upload), Inventory Health, and Settings.
*   **Header Area:** Category selector (Grocery, Household, Hobbies), Item selector, and a Date Range / Forecast Horizon dropdown.
*   **Primary Content Grid (The "Decision Hub"):**
    *   **Top Row:** High-impact metric cards summarising the Business Advisor metrics (Total Expected Demand, Total Procurement Cost, Critical Stock Alerts).
    *   **Center Stage (Dual Visualization):**
        *   *Left:* **The Demand Forecast Chart.** A large time-series graph displaying the 30-day historical footprint seamlessly connecting into the N-day forecast trajectory, shaded with the model's upper and lower uncertainty bounds.
        *   *Right:* **The XAI Explainability Panel.** Placed identically adjacent to the forecast graph. Selecting a predicted day on the left chart dynamically updates the right panel to show a contiguous heatmap or bar chart detailing the feature importance mapping for that specific prediction (e.g., highlighting that 'is_event_day' and a spike in 'price' 2 weeks ago heavily influenced tomorrow's forecast).
    *   **Bottom Section:** The **Conversational AI / Business Advisor Panel** offering the structured narrative insights, alongside data tables for individual product reorder actions.

## 2. Component Structure

The frontend application should be structured modularly as follows:

*   **`Layout/`**
    *   `Sidebar.jsx`: Global navigation and system status indicators.
    *   `Header.jsx`: Contains the `CategorySelect` and `ItemSelect` dropdowns, alongside global parameters (e.g., Forecast Horizon).
*   **`Dashboard/`**
    *   `KpiWidgetGroup.jsx`: Container for the high-level summary metrics.
    *   `ForecastModule.jsx`: A unified container managing the synchronized state between the charting components.
        *   `LSTMForecastChart.jsx`: Renders the time-series line chart with confidence intervals.
        *   `XAIExplainabilityPanel.jsx`: Visualizes the 30-day lookback feature importance.
*   **`DataAcquisition/`**
    *   `CSVUploadModal.jsx`: Interface for managers to drop the multivariate sales `.csv` to hit the `/predict_custom` route.
*   **`Advisor/`**
    *   `BusinessAdvisorPanel.jsx`: Layout container for text-based insights.
    *   `InsightCard.jsx`: Reusable card component for the 5 distinct insight categories.
    *   `ChatInterface.jsx` (Future): Placeholder container ready for the interactive NLP input.

## 3. API Integration Mapping

*   **`GET /health`**
    *   *Consumed by:* `Sidebar.jsx` (on mount) to show the "System Active" green dot and verify API connectivity.
*   **`GET /items/{category}`**
    *   *Consumed by:* `Header.jsx`. Triggered whenever the `CategorySelect` dropdown changes. Populates the dependent `ItemSelect` dropdown.
*   **`POST /predict`**
    *   *Consumed by:* `ForecastModule.jsx`. Triggered when a manager selects a new item from the `ItemSelect` dropdown. It fetches the `predictions`, `lower_bound`, and `upper_bound` for the `LSTMForecastChart.jsx`, and fetches the corresponding feature importance array for the `XAIExplainabilityPanel.jsx`.
*   **`POST /predict_custom`**
    *   *Consumed by:* `CSVUploadModal.jsx`. Upon successful user upload, this sends the file contents. The response feeds directly into a global application state (like Zustand/Redux) forcing the `KpiWidgetGroup`, `ForecastModule`, and `BusinessAdvisorPanel` to re-render with the custom data rather than static cached items.

## 4. Data Visualization Recommendations

Handling time-series data with dynamic confidence intervals and adjacent XAI heatmaps requires a robust, high-performance charting library.

1.  **Apache ECharts (via `echarts-for-react`) - Highly Recommended**
    *   *Why:* Superb performance with large datasets. Excellent out-of-the-box support for shaded confidence intervals (`line` series combined with `areastyle` between upper/lower bounds). Crucially, its customizability makes it easy to build a heatmapped sequence bar chart for the 30-day XAI contiguous mapping.
2.  **Plotly.js (via `react-plotly.js`)**
    *   *Why:* Native Python familiarity. If the current prototype uses Plotly in Streamlit, migrating the exact JSON configurations over to React/Vue is incredibly trivial. Excellent scientific plotting capabilities.
3.  **Chart.js (via `react-chartjs-2`)**
    *   *Why:* Easiest to implement for basic layouts, but drawing dynamic confidence intervals requires custom plugin logic, making it less ideal for the `upper_bound`/`lower_bound` requirements.

## 5. Conversational AI UI Strategy

### Current Layout (Rule-Based Insights)
The `BusinessAdvisorPanel` should act as the concluding "Executive Summary" at the bottom of the dashboard.
*   **Visual Structure:** A clean, masonry-style grid of 5 distinct `InsightCard` components.
*   The **Executive Summary** card spans full width at the top, styled prominently (e.g., slight background tint, larger font).
*   Below it, a 2x2 grid housing the **Demand**, **Category**, **Procurement**, and **Inventory** insights. Each card uses specific icons (📈, 📦, 🛒) to ensure quick scanning by managers.

### Future Transition (Interactive NLP ChatBox)
To seamlessly integrate the future Gemini NLP ChatBox:
*   Add a floating **"Ask the AI" Fab (Floating Action Button)** in the bottom right corner of the application.
*   When clicked, it slides out a side-panel drawer. This drawer initially populates its chat history with the 5 current rule-based insights as the "Greeting" from the bot.
*   The input box at the bottom of the drawer will eventually connect to the Gemini backend, allowing the manager to type follow-up questions natively within the context of the currently visible dashboard data.

## 6. Step-by-Step Implementation Plan

### Phase 1: Scaffold & Static Layout
1.  Initialize frontend framework (React/Next.js).
2.  Build the static `Layout` components (`Sidebar`, `Header`) using hardcoded dummy categories (Food, Household).
3.  Implement the `BusinessAdvisorPanel` layout using mock strings for the 5 insight categories.

### Phase 2: Fundamental API Connectivity
4.  Implement API service functions using `fetch` or `axios`.
5.  Connect `Header`: On mount, hit `GET /health`. On category change, hit `GET /items/{category}` to dynamically populate the item dropdowns.
6.  Connect `ForecastModule`: On item selection, trigger `POST /predict`. Log the raw JSON output to the browser console to verify connection.

### Phase 3: Data Visualization
7.  Integrate the chosen charting library (e.g., ECharts).
8.  Construct the `LSTMForecastChart.jsx`. Map the 30-day historical data points alongside the `predictions`, shaded by the `lower_bound` and `upper_bound` arrays returned from `/predict`.
9.  Construct the `XAIExplainabilityPanel.jsx`. Map the contiguous importance arrays to a stylized bar/heatmap visual.

### Phase 4: Custom Workflows & Polish
10. Build the `CSVUploadModal.jsx`. Parse the CSV locally (e.g., via `PapaParse`), validate the schema, and dispatch the array to `POST /predict_custom`.
11. Implement global state management (so uploading the CSV globally updates the charts and the Business Advisor text simultaneously).
