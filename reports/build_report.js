// Build the final report as a .docx
// Clean editorial layout - charcoal serif headings, calibri body, no color chrome.

const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, PageBreak,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, TabStopType,
  TabStopPosition, LevelFormat, PageOrientation, Header, Footer, PageNumber,
  ExternalHyperlink, PositionalTab, PositionalTabAlignment, PositionalTabLeader,
} = require('docx');

// --------------------------------------------------------------------------
// Design tokens
// --------------------------------------------------------------------------
const INK           = "1a1a1a";
const CHARCOAL      = "2c2c2c";
const HAIRLINE      = "c4c4c4";
const HEADER_BG     = "f5f5f5";
const SERIF         = "Georgia";
const BODY          = "Calibri";
const MONO          = "Consolas";

// Half-points for docx font sizes (24 = 12pt)
const SZ_TITLE      = 56;   // 28pt
const SZ_SUB        = 22;   // 11pt
const SZ_H1         = 32;   // 16pt
const SZ_H2         = 26;   // 13pt
const SZ_BODY       = 22;   // 11pt
const SZ_SMALL      = 20;   // 10pt

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 300 },
    children: [
      new TextRun({
        text, font: BODY, size: SZ_BODY, color: INK,
        italics: opts.italics, bold: opts.bold,
      }),
    ],
  });
}

// Body paragraph with mixed inline runs (accepts an array of TextRun)
function para(runs, spacingAfter = 160) {
  return new Paragraph({
    spacing: { after: spacingAfter, line: 300 },
    children: runs,
  });
}

function run(text, opts = {}) {
  return new TextRun({
    text, font: BODY, size: SZ_BODY, color: INK,
    bold: opts.bold, italics: opts.italics,
    ...opts,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 480, after: 240 },
    border: { bottom: { color: HAIRLINE, style: BorderStyle.SINGLE, size: 6, space: 4 } },
    children: [
      new TextRun({ text, font: SERIF, size: SZ_H1, color: CHARCOAL, bold: false }),
    ],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 320, after: 140 },
    children: [
      new TextRun({ text, font: SERIF, size: SZ_H2, color: CHARCOAL, bold: false }),
    ],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 100, line: 300 },
    children: [ new TextRun({ text, font: BODY, size: SZ_BODY, color: INK }) ],
  });
}

function bulletRuns(runs) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 100, line: 300 },
    children: runs,
  });
}

function link(text, url) {
  return new ExternalHyperlink({
    link: url,
    children: [
      new TextRun({
        text, font: BODY, size: SZ_BODY, color: "1a4d80", underline: {},
      }),
    ],
  });
}

// Simple table cell
function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: HEADER_BG, color: "auto" } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [
      new Paragraph({
        alignment: opts.align || AlignmentType.LEFT,
        children: [
          new TextRun({
            text: String(text),
            font: opts.mono ? MONO : BODY,
            size: SZ_SMALL,
            color: INK,
            bold: opts.header || opts.bold,
          }),
        ],
      }),
    ],
  });
}

// --------------------------------------------------------------------------
// Model comparison table
// --------------------------------------------------------------------------
const MODEL_ROWS = [
  ["24 h", "ridge",         "18.02", "13.41",  "0.20"],
  ["24 h", "random_forest", "18.36", "13.51",  "0.17"],
  ["24 h", "persistence",   "18.63", "12.71",  "0.15"],
  ["24 h", "xgboost",       "19.44", "14.74",  "0.07"],
  ["24 h", "lightgbm",      "19.76", "15.02",  "0.04"],
  ["24 h", "lstm (torch)",  "22.51", "17.31",  "-0.24"],
  ["48 h", "ridge",         "19.94", "14.83",  "0.08"],
  ["48 h", "persistence",   "23.08", "16.84",  "-0.23"],
  ["48 h", "random_forest", "23.69", "18.73",  "-0.30"],
  ["48 h", "xgboost",       "25.53", "20.16",  "-0.51"],
  ["48 h", "lightgbm",      "26.39", "21.22",  "-0.61"],
  ["48 h", "lstm (torch)",  "35.16", "29.80",  "-1.86"],
  ["72 h", "ridge",         "21.74", "16.67",  "-0.07"],
  ["72 h", "persistence",   "22.94", "17.30",  "-0.19"],
  ["72 h", "random_forest", "24.94", "19.62",  "-0.41"],
  ["72 h", "lightgbm",      "26.70", "21.56",  "-0.61"],
  ["72 h", "xgboost",       "27.14", "21.80",  "-0.66"],
  ["72 h", "lstm (torch)",  "45.81", "38.52",  "-3.74"],
];

const COL_WIDTHS = [1200, 2400, 1800, 1800, 1800]; // sum = 9000
function modelTable() {
  const border = { style: BorderStyle.SINGLE, size: 4, color: HAIRLINE };
  const rows = [];
  // header
  rows.push(new TableRow({
    tableHeader: true,
    children: [
      cell("Horizon",   { width: COL_WIDTHS[0], header: true }),
      cell("Model",     { width: COL_WIDTHS[1], header: true }),
      cell("Test RMSE", { width: COL_WIDTHS[2], header: true, align: AlignmentType.RIGHT }),
      cell("Test MAE",  { width: COL_WIDTHS[3], header: true, align: AlignmentType.RIGHT }),
      cell("Test R²",   { width: COL_WIDTHS[4], header: true, align: AlignmentType.RIGHT }),
    ],
  }));
  for (const r of MODEL_ROWS) {
    const isBest = r[1] === "ridge"; // ridge wins every horizon
    rows.push(new TableRow({
      children: [
        cell(r[0], { width: COL_WIDTHS[0], bold: isBest }),
        cell(r[1], { width: COL_WIDTHS[1], bold: isBest, mono: true }),
        cell(r[2], { width: COL_WIDTHS[2], align: AlignmentType.RIGHT, bold: isBest, mono: true }),
        cell(r[3], { width: COL_WIDTHS[3], align: AlignmentType.RIGHT, bold: isBest, mono: true }),
        cell(r[4], { width: COL_WIDTHS[4], align: AlignmentType.RIGHT, bold: isBest, mono: true }),
      ],
    }));
  }
  return new Table({
    columnWidths: COL_WIDTHS,
    rows,
    borders: {
      top:              { ...border, size: 6 },
      bottom:           { ...border, size: 6 },
      left:             { style: BorderStyle.NONE, size: 0, color: "auto" },
      right:            { style: BorderStyle.NONE, size: 0, color: "auto" },
      insideHorizontal: border,
      insideVertical:   { style: BorderStyle.NONE, size: 0, color: "auto" },
    },
  });
}

// --------------------------------------------------------------------------
// Content
// --------------------------------------------------------------------------
const children = [];

// Title page
children.push(new Paragraph({ spacing: { before: 2400 }, children: [ new TextRun({ text: "" }) ] }));
children.push(new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { after: 120 },
  children: [ new TextRun({ text: "Pearls AQI Predictor", font: SERIF, size: SZ_TITLE, color: CHARCOAL }) ],
}));
children.push(new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { after: 480 },
  children: [
    new TextRun({
      text: "A three-day air quality forecast for Lahore, built end to end on a free serverless stack.",
      font: SERIF, size: 26, color: "6b6b6b", italics: true,
    }),
  ],
}));
children.push(new Paragraph({
  spacing: { after: 60, before: 2000 },
  children: [ new TextRun({ text: "Bilal Asif", font: BODY, size: 24, color: INK }) ],
}));
children.push(new Paragraph({
  spacing: { after: 60 },
  children: [ new TextRun({ text: "September 2026", font: BODY, size: 22, color: "6b6b6b" }) ],
}));
children.push(new Paragraph({
  children: [ new TextRun({ text: "Pearls AQI Predictor challenge submission", font: BODY, size: 22, color: "6b6b6b" }) ],
}));
children.push(new Paragraph({ children: [ new PageBreak() ] }));

// --- 1. Introduction ---
children.push(h1("1. Introduction"));
children.push(body(
  "Lahore has some of the worst air quality in the world. On the day I finished this project, the city was sitting at an AQI of around 156, which is firmly in the \"unhealthy\" range. Knowing whether the air is going to get better or worse in the next few days matters if you have breathing issues, want to plan outdoor time, or run any kind of event outdoors."
));
children.push(body(
  "This project is a serverless machine learning system that forecasts the AQI in Lahore three days ahead. It runs entirely on free tiers. Data comes from the Open-Meteo air quality API, features and models are stored on Hopsworks, GitHub Actions handles the hourly and daily scheduling, and Streamlit Community Cloud serves the dashboard."
));
children.push(para([
  run("The live dashboard is at "),
  link("pearls-aqi-lahore.streamlit.app", "https://pearls-aqi-lahore.streamlit.app"),
  run(" and the source code is on GitHub at "),
  link("github.com/bilalasif878-wq/pearls-aqi-predictor", "https://github.com/bilalasif878-wq/pearls-aqi-predictor"),
  run("."),
]));

// --- 2. Data sources ---
children.push(h1("2. Data sources"));
children.push(body(
  "I started with the intention of using AQICN, but discovered pretty quickly that the Lahore US Embassy station on AQICN (which is the only Pakistani station indexed there) stopped reporting in February 2025. The four Pakistan stations on AQICN are all US diplomatic mission stations and all of them appear to be offline. So the \"real time\" reading you get from AQICN for Lahore is over 18 months old."
));
children.push(body(
  "I switched to Open-Meteo instead. Open-Meteo's air-quality endpoint returns the Copernicus CAMS global reanalysis, which is a physics-based model that assimilates satellite observations and ground stations worldwide. It is not tied to any single sensor being online, updates hourly, and gives me PM2.5, PM10, ozone, nitrogen dioxide, sulphur dioxide, carbon monoxide, and the calculated US AQI. For weather I use Open-Meteo's archive and forecast endpoints, which give me temperature, humidity, dew point, precipitation, wind speed, wind direction, surface pressure, and cloud cover."
));
children.push(body(
  "Using one source for both live data and historical backfill turned out to be cleaner than my original two-source design. Same schema, no timestamp alignment issues, and the historical data goes back years so backfill was straightforward."
));

// --- 3. Architecture ---
children.push(h1("3. Architecture"));
children.push(body("The project is organised as four independent pipelines that communicate only through Hopsworks:"));
children.push(bulletRuns([
  run("Feature pipeline. ", { bold: true }),
  run("Pulls the latest hour of air quality and weather from Open-Meteo and writes it to the Hopsworks feature group. Runs every hour on GitHub Actions."),
]));
children.push(bulletRuns([
  run("Backfill. ", { bold: true }),
  run("The same feature computation done for the last 400 days in bulk. Ran once, produced 9,624 hourly rows."),
]));
children.push(bulletRuns([
  run("Training pipeline. ", { bold: true }),
  run("Reads the full feature history from Hopsworks, engineers lag, rolling and time features, trains five candidate models per forecast horizon, and registers the best model per horizon in the Hopsworks model registry. Runs daily."),
]));
children.push(bulletRuns([
  run("Dashboard. ", { bold: true }),
  run("Reads the latest features and the current best model from Hopsworks and shows the 3-day forecast with health alerts and SHAP feature attributions."),
]));
children.push(body(
  "Splitting things this way means each pipeline can fail independently without breaking the others. The feature pipeline missing an hour just means the training pipeline sees one fewer row, and the dashboard is unaffected as long as anything got written that day."
));

// --- 4. Feature engineering ---
children.push(h1("4. Feature engineering"));
children.push(body(
  "The raw hourly row from Open-Meteo has around eight pollutant and weather variables. Feeding those directly to a model gets you almost nothing, so most of the work here is turning that thin input into features the model can actually learn from."
));
children.push(body("I added three categories of engineered features:"));
children.push(bulletRuns([
  run("Time features. ", { bold: true }),
  run("Hour of day, day of week, day of year, month, is_weekend, plus sine/cosine cyclical encodings of hour, day-of-week, and month. The cyclical encoding matters because hour 23 and hour 0 are neighbours, and Sunday and Monday are neighbours, but the model would not know that from raw integers."),
]));
children.push(bulletRuns([
  run("Lag features. ", { bold: true }),
  run("For every pollutant and key weather variable, the value 1, 3, 6, 12, 24 and 48 hours ago. Pollution is highly autocorrelated so these lags carry most of the signal."),
]));
children.push(bulletRuns([
  run("Rolling features. ", { bold: true }),
  run("6, 12 and 24-hour moving averages, maxes and standard deviations, computed on shifted values so they never include the current hour. Including the current hour would be temporal leakage."),
]));
children.push(body(
  "There are also two derived features: aqi_change_3h and aqi_change_24h as trend indicators, and u/v wind components computed from wind speed and direction. In total the engineered dataframe has around 145 columns."
));
children.push(body(
  "I wrote unit tests for the feature builder to catch leakage. The tests verify that lag_1h really is the previous hour's value, that rolling means do not include the current hour, and that targets are shifted correctly. All five tests pass on synthetic data."
));

// --- 5. Models and results ---
children.push(h1("5. Models and results"));
children.push(body(
  "I trained six candidate models for each of three forecast horizons (24, 48, and 72 hours ahead):"
));
children.push(bulletRuns([
  run("Persistence baseline. ", { bold: true }),
  run("Predicts that the AQI in N hours will be the same as it is right now. This is the \"did the model actually learn anything\" benchmark."),
]));
children.push(bullet("Ridge regression with feature scaling."));
children.push(bullet("Random Forest with 300 trees."));
children.push(bullet("XGBoost with 500 boosting rounds."));
children.push(bullet("LightGBM with 500 boosting rounds."));
children.push(bulletRuns([
  run("PyTorch LSTM. ", { bold: true }),
  run("A small 1-layer LSTM (hidden size 64) fed a length-7 sequence assembled from the existing lag features (values at t-48, t-24, t-12, t-6, t-3, t-1 and t) for eleven pollutant and weather variables, followed by an MLP head. Trained for 15 epochs on Apple Silicon's MPS backend."),
]));
children.push(body(
  "The training uses a time-based split: about 340 days for training, then a 30-day validation window, then the most recent 30 days as the test set. Random splits would leak future information into the training set, which is a common pitfall in time-series problems."
));
children.push(body("Test-set performance for each model (sorted best to worst within each horizon):"));
children.push(modelTable());
children.push(new Paragraph({ spacing: { after: 200 } }));
children.push(body(
  "Ridge won every horizon. This was a bit surprising at first because I expected the boosted trees or the LSTM to win. Reading back through the data I think what happened is that the features are well-conditioned linear signals (lags and rolling means correlate almost linearly with future AQI), and the fancier models had less room to add value than I expected. The boosted trees were overfitting to seasonal patterns in the training set that did not generalise, and the 30-day test window can be a real distribution shift from the training window ending 30 days earlier."
));
children.push(body(
  "The LSTM in particular was disappointing. It came last on every horizon and got dramatically worse as the horizon increased. I think there are a few reasons for that. The sequence I fed it was very sparse (only six historical timesteps: 48, 24, 12, 6, 3 and 1 hours back), so the LSTM machinery was not really operating over a proper hourly sequence. The training set is also small for a deep model (about 8,000 rows), and I only trained for 15 epochs. The takeaway is a common one for tabular time-series problems with modest amounts of data: well-engineered linear or tree models often beat deep learning, and the deep model becomes worthwhile only when you have longer sequences, more data, or richer features that a linear model cannot compose on its own."
));
children.push(body(
"The R² drops off sharply with horizon, which is honest, not a bug. Predicting AQI 72 hours ahead really is much harder than predicting 24 hours ahead. The Ridge 72h R² is basically zero, which means the model is barely better than just predicting the mean. But its RMSE is still better than the persistence baseline, so it is doing something."
));
children.push(body(
  "All three winning models (Ridge for 24h, 48h, and 72h) are registered in the Hopsworks model registry as separate named models. The dashboard pulls the best-by-RMSE version at inference time."
));

// --- 6. SHAP ---
children.push(h1("6. Feature attribution with SHAP"));
children.push(body(
  "The SHAP panel in the dashboard shows the top contributing features to the most recent 24-hour prediction using a waterfall plot. Because the winning model is a scikit-learn Pipeline (StandardScaler plus Ridge), I had to use LinearExplainer on the final regressor step after transforming inputs through the scaler. The default TreeExplainer does not work for Pipelines."
));
children.push(body(
  "The plot typically shows aqi_lag_1h and aqi_lag_3h as the top drivers, which makes sense: the strongest signal for tomorrow's AQI is right-now's AQI. Wind speed and temperature also feature prominently, which lines up with the physics (wind disperses pollutants, warm and dry conditions concentrate them)."
));

// --- 7. Deployment ---
children.push(h1("7. Deployment and automation"));
children.push(para([
  run("Feature pipeline ", { bold: true }),
  run("runs on GitHub Actions every hour at HH:05, which gives Open-Meteo enough time to publish the current hour's data before we fetch it. Each run takes about 90 seconds."),
]));
children.push(para([
  run("Training pipeline ", { bold: true }),
  run("runs daily at 02:30 UTC (07:30 in Lahore). Each run takes 5 to 10 minutes because Random Forest and the boosters are not small models. It reads the full feature history, retrains all 15 model/horizon combinations, picks the best per horizon, and registers them."),
]));
children.push(para([
  run("Dashboard ", { bold: true }),
  run("is deployed on Streamlit Community Cloud at "),
  link("pearls-aqi-lahore.streamlit.app", "https://pearls-aqi-lahore.streamlit.app"),
  run(". It is a free public URL that Streamlit rebuilds whenever I push to the main branch of the GitHub repo. The Hopsworks credentials are stored in Streamlit's secrets, not in the repo."),
]));
children.push(para([
  run("FastAPI endpoint ", { bold: true }),
  run("is also available in the code (app/api.py) but not deployed publicly. It returns the same forecast as JSON and would be the entry point if I wanted to build a mobile app or embed the prediction in another site."),
]));

// --- 8. Challenges ---
children.push(h1("8. What was harder than expected"));
children.push(body("A few things bit me during this project:"));
children.push(bulletRuns([
  run("AQICN Lahore station being offline. ", { bold: true }),
  run("I lost a couple of hours going back and forth trying different station IDs before figuring out that the whole Pakistan-side coverage on AQICN is offline. Pivoting to Open-Meteo was actually a better design in the end but I would have gotten there faster if I had checked coverage first."),
]));
children.push(bulletRuns([
  run("Hopsworks project name collision. ", { bold: true }),
  run("Project names in Hopsworks are globally unique across all users on the free tier. My first attempt at \"pearls\" gave a 409 Conflict because someone else already had it, but the UI just hangs instead of showing the error. I only figured this out from the browser console. Naming the project bilal_aqi_lhr (with my name in it) worked."),
]));
children.push(bulletRuns([
  run("Python 3.14 compatibility. ", { bold: true }),
  run("The hopsworks library has not published wheels for Python 3.14 yet, so I rebuilt the environment on Python 3.12 both locally and on Streamlit Cloud."),
]));
children.push(bulletRuns([
  run("Feature group time-travel format. ", { bold: true }),
  run("New Hopsworks feature groups default to Delta which needs an extra library. Switching to HUDI avoided that dependency."),
]));
children.push(bulletRuns([
  run("libomp missing on my Mac. ", { bold: true }),
  run("XGBoost and LightGBM would not load until I ran brew install libomp."),
]));
children.push(bulletRuns([
  run("A .gitignore rule that matched too much. ", { bold: true }),
  run("The rule data/ was also matching src/data/, so my API client files quietly were not committed. I only found out when the first GitHub Actions run failed with ModuleNotFoundError. The fix was to change the rule to /data/ so it only matches the top level."),
]));

// --- 9. Limitations ---
children.push(h1("9. Limitations and what I would do next"));
children.push(body("Some honest limitations of what I built:"));
children.push(bulletRuns([
  run("72-hour predictions are barely better than baseline. ", { bold: true }),
  run("This is expected but not great. To really improve this I would need more history (I only have 400 days), a weather-forecast-based feature set (Open-Meteo forecasts future weather too), and probably a proper deep-learning model on a bigger dataset."),
]));
children.push(bulletRuns([
  run("The models are only trained on one city. ", { bold: true }),
  run("They would need to be retrained separately for each city I add."),
]));
children.push(bulletRuns([
  run("No uncertainty estimates. ", { bold: true }),
  run("I show a point forecast but not a confidence interval. Adding a quantile-regression version or bootstrapping would tell users how much to trust each prediction."),
]));
children.push(bulletRuns([
  run("Retraining is naive. ", { bold: true }),
  run("Every day it does a full retrain from scratch. In a real system I would use online learning or at least warm-start."),
]));
children.push(bulletRuns([
  run("No monitoring. ", { bold: true }),
  run("I do not currently track prediction accuracy over time or alert if a model degrades. A next step would be to write actual predictions to Hopsworks alongside the eventual truth values and compute rolling accuracy."),
]));

// --- 10. Links ---
children.push(h1("10. Links and references"));
const links = [
  ["Live dashboard",                "https://pearls-aqi-lahore.streamlit.app"],
  ["Source code (GitHub)",          "https://github.com/bilalasif878-wq/pearls-aqi-predictor"],
  ["Open-Meteo air quality API",    "https://open-meteo.com/en/docs/air-quality-api"],
  ["Open-Meteo weather archive",    "https://open-meteo.com/en/docs/historical-weather-api"],
  ["Hopsworks",                     "https://www.hopsworks.ai"],
  ["SHAP documentation",            "https://shap.readthedocs.io"],
];
for (const [label, url] of links) {
  children.push(para([
    run(label + ". "),
    link(url, url),
  ], 100));
}

// --------------------------------------------------------------------------
// Document
// --------------------------------------------------------------------------
const doc = new Document({
  creator: "Bilal Asif",
  title: "Pearls AQI Predictor",
  description: "Three-day AQI forecast for Lahore.",
  styles: {
    default: {
      document: { run: { font: BODY, size: SZ_BODY, color: INK } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true, run: { font: SERIF, size: SZ_H1, color: CHARCOAL } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        quickFormat: true, run: { font: SERIF, size: SZ_H2, color: CHARCOAL } },
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 480, hanging: 260 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [ new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [
            new TextRun({ text: "Bilal Asif  .  Pearls AQI Predictor  .  ", font: BODY, size: 18, color: "8a8a8a" }),
            new TextRun({ children: [PageNumber.CURRENT], font: BODY, size: 18, color: "8a8a8a" }),
          ],
        }) ],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const outPath = path.resolve(__dirname, "../Pearls_AQI_Predictor_Report.docx");
  fs.writeFileSync(outPath, buf);
  console.log("wrote " + outPath + " (" + buf.length + " bytes)");
});
