const { useState, useEffect, useMemo, useRef } = React;
const createElement = React.createElement;

const STATS_LABELS = [
  { key: "points", label: "Points" },
  { key: "per", label: "PER" },
];

const formatStat = (key, stats = {}) => {
  const val = stats[key];
  if (val == null) return "--";
  if (typeof val === "number") return Number(val).toFixed(1);
  const num = Number(val);
  return Number.isNaN(num) ? val : num.toFixed(1);
};

// XGBoost JSON inference (array-based trees)
const predictTree = (tree, features) => {
  let node = 0;
  const left = tree.left_children;
  const right = tree.right_children;
  const splitIdx = tree.split_indices;
  const splitCond = tree.split_conditions;
  const defaultLeft = tree.default_left;

  while (left[node] !== -1 && right[node] !== -1) {
    const fid = Number(splitIdx[node]);
    const fval = fid < features.length ? features[fid] : 0;
    const cond = Number(splitCond[node]);
    const isMissing = fval === null || Number.isNaN(fval);
    const goLeft = isMissing ? !!Number(defaultLeft[node]) : fval < cond;
    node = goLeft ? Number(left[node]) : Number(right[node]);
  }
  return Number(splitCond[node]);
};

const predictModel = (model, features) => {
  const trees = model.learner.gradient_booster.model.trees;
  const rawBase = model.learner.learner_model_param.base_score;
  let base;
  if (Array.isArray(rawBase)) {
    base = Number(rawBase[0]);
  } else if (typeof rawBase === "string") {
    base = Number(rawBase.replace(/[\[\]\\s]/g, ""));
  } else {
    base = Number(rawBase);
  }
  if (Number.isNaN(base)) base = 0;
  let score = base;
  for (const tree of trees) {
    score += predictTree(tree, features);
  }
  return score;
};

const Tile = ({ player, history, year, index, onDragStart, onDrop, slotClasses }) => {
  const stats = player?.stats || {};
  const showYear = player?.season && player.season !== year;
  const labelBase = player?.name || `Slot ${player?.slot ?? index + 1} (open)`;
  const label = showYear ? `${labelBase} (${player.season})` : labelBase;

  return createElement(
    "div",
    {
      className: "player-tile",
      draggable: true,
      onDragStart: () => onDragStart(index),
      onDragOver: (event) => event.preventDefault(),
      onDrop: () => onDrop(index),
    },
    createElement(
      "div",
      { className: "player-head" },
      createElement("span", { className: (slotClasses || ["player-slot"]).join(" ") }, `#${player?.slot ?? index + 1}`),
      createElement("h3", { className: "player-name" }, label)
    ),
    createElement(
      "div",
      { className: "player-stats" },
      ...STATS_LABELS.map(({ key, label }) =>
        createElement(
          "div",
          { key, className: "stat" },
          createElement("span", { className: "stat-label" }, label),
          createElement(
            "span",
            { className: "stat-value" },
            formatStat(key, stats)
          )
        )
      )
    )
  );
};

const App = () => {
  const [data, setData] = useState(null);
  const [selectedTeam, setSelectedTeam] = useState("");
  const [selectedYear, setSelectedYear] = useState(null);
  const [playerOrder, setPlayerOrder] = useState([]);
  const [baselineRoster, setBaselineRoster] = useState([]);
  const dragIndex = useRef(null);
  const [dragOver, setDragOver] = useState(null);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [swapSelection, setSwapSelection] = useState("");
  const [featureCols, setFeatureCols] = useState([]);
  const [modelJson, setModelJson] = useState(null);
  const [predictedWins, setPredictedWins] = useState(null);

  useEffect(() => {
    fetch("data/rosters.json")
      .then((response) => response.json())
      .then((payload) => {
        setData(payload);
        setSelectedTeam(payload.teams[0] ?? "");
        setSelectedYear(payload.years[payload.years.length - 1] ?? null);
      })
      .catch(console.error);

    fetch("data/feature_cols_old_era.json")
      .then((r) => r.json())
      .then(setFeatureCols)
      .catch(console.error);

    fetch("data/xgb_model_old_era.json")
      .then((r) => r.json())
      .then(setModelJson)
      .catch(console.error);
  }, []);

  const roster = useMemo(() => {
    if (!data || !selectedTeam || selectedYear == null) {
      return null;
    }
    return data.rosters.find(
      (entry) => entry.team === selectedTeam && entry.season === selectedYear
    );
  }, [data, selectedTeam, selectedYear]);

  const teamValid = useMemo(
    () => (data?.teams ?? []).includes(selectedTeam),
    [data?.teams, selectedTeam]
  );
  const yearValid = useMemo(
    () => (data?.years ?? []).includes(selectedYear),
    [data?.years, selectedYear]
  );

  useEffect(() => {
    if (!roster) {
      setPlayerOrder([]);
      setBaselineRoster([]);
      return;
    }
    const next = roster.players.map((player) => ({
      ...player,
      season: roster.season,
    }));
    setPlayerOrder(next);
    setBaselineRoster(next);
    setPredictedWins(null);
  }, [roster?.team, roster?.season]);

  const handleDragStart = (index) => {
    dragIndex.current = index;
    setSelectedPlayer(null);
  };

  const handleDrop = (targetIndex) => {
    if (dragIndex.current == null || !playerOrder.length) {
      return;
    }
    const updated = [...playerOrder];
    const [moved] = updated.splice(dragIndex.current, 1);
    updated.splice(targetIndex, 0, moved);
    // re-slot after reorder so p1 is top-left, p2 is top-right, etc.
    const reSlotted = updated.map((player, idx) => ({
      ...player,
      slot: idx + 1,
    }));
    setPlayerOrder(reSlotted);
    setPredictedWins(null);
    dragIndex.current = null;
    setDragOver(null);
  };

  const handleDragEnter = (index) => setDragOver(index);
  const handleDragLeave = () => setDragOver(null);

  const expectedWins =
    predictedWins != null
      ? predictedWins.toFixed(2)
      : roster?.regSeasonWins != null
      ? roster.regSeasonWins.toFixed(1)
      : "--";
  const actualWins =
    roster?.regSeasonWins != null ? roster.regSeasonWins.toFixed(1) : "--";

  const handleRunModel = () => {
    if (!modelJson || !featureCols.length) {
      console.warn("Model or feature columns not loaded yet.");
      return;
    }

    // Build feature vector in model order
    const slotMap = {};
    playerOrder.forEach((p) => {
      slotMap[p.slot] = p;
    });

    const feats = featureCols.map((fname) => {
      const match = fname.match(/^p(\d+)_(.+)$/);
      if (!match) return 0;
      const slot = Number(match[1]);
      const stat = match[2];
      const player = slotMap[slot];
      if (!player || !player.stats) return 0;
      const val = player.stats[stat];
      const num = val == null ? 0 : Number(val);
      return Number.isNaN(num) ? 0 : num;
    });

    const pred = predictModel(modelJson, feats);
    setPredictedWins(pred);
    console.log("Run model – expected wins:", pred);
  };

  // Auto-run model when roster changes and model is ready
  useEffect(() => {
    if (!modelJson || !featureCols.length || !playerOrder.length) return;
    handleRunModel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playerOrder, modelJson, featureCols]);

  const handleSelectPlayer = (player, history) => {
    setSelectedPlayer({
      ...player,
      history,
    });
    setSwapName("");
    setSwapYear(null);
  };

  const playerYearOptions = useMemo(() => {
    if (!data) return [];
    const out = [];
    Object.entries(data.playerHistories || {}).forEach(([name, hist]) => {
      hist.forEach((h) => {
        out.push(`${name} | ${h.season} | ${h.team}`);
      });
    });
    return out;
  }, [data]);

  const handleSwap = () => {
    if (!selectedPlayer || !swapSelection || !data) return;
    const parts = swapSelection.split("|").map((s) => s.trim());
    if (parts.length < 2) return;
    const swapName = parts[0];
    const swapYear = Number(parts[1]);
    const swapTeam = parts[2] || null;

    const hist = data.playerHistories[swapName];
    if (!hist) return;
    const entry =
      hist.find((h) => h.season === swapYear && (!swapTeam || h.team === swapTeam)) ||
      hist.find((h) => h.season === swapYear) ||
      hist[hist.length - 1];
    const stats = entry?.stats || {};

    const updated = playerOrder.map((p) =>
      p.slot === selectedPlayer.slot
        ? { ...p, name: swapName, stats, slot: p.slot, season: swapYear }
        : p
    );
    setPlayerOrder(updated);
    setPredictedWins(null);
    setSelectedPlayer(null);
    setSwapSelection("");
  };

  const teamOptions =
    data?.teams.map((team) => createElement("option", { key: team, value: team }, team)) ||
    [];
  const yearOptions =
    data?.years.map((year) => createElement("option", { key: year, value: year }, year)) ||
    [];
  const playerTiles = playerOrder.map((player, index) => {
    const classNames = ["player-tile"];
    if (dragIndex.current === index) classNames.push("dragging");
    if (dragOver === index) classNames.push("drag-target");
    const baselineEntry = baselineRoster.find((p) => p.slot === player.slot);
    const onTeam =
      baselineEntry &&
      baselineEntry.name === player.name &&
      baselineEntry.season === player.season;
    if (onTeam) classNames.push("player-on-team");
    else classNames.push("player-off-team");
    const slotClasses = ["player-slot"];
    if (onTeam) slotClasses.push("player-slot-on-team");

    return createElement(
      "div",
      {
        key: `${player.slot}-${player.name}-${index}`,
        className: classNames.join(" "),
        draggable: true,
        onDragStart: () => handleDragStart(index),
        onDragOver: (event) => event.preventDefault(),
        onDrop: () => handleDrop(index),
        onDragEnter: () => handleDragEnter(index),
        onDragLeave: handleDragLeave,
        onClick: () =>
          handleSelectPlayer(player, data?.playerHistories?.[player.name] ?? []),
      },
      createElement(Tile, {
        player,
        history: data?.playerHistories?.[player.name] ?? [],
        year: selectedYear,
        index,
        onDragStart: handleDragStart,
        onDrop: handleDrop,
        slotClasses,
      })
    );
  });

  return createElement(
    "div",
    { className: "app-shell" },
    createElement("h1", { className: "page-title" }, "Machine Learning to Analyze Team-Player Win Returns"),
    createElement(
      "div",
      { className: "layout" },
      // left column
      createElement(
        "div",
        null,
        createElement(
          "header",
          { className: "top-panel" },
          createElement(
            "div",
            null,
            createElement("p", { className: "label" }, "Team"),
            createElement(
              "select",
              {
                value: selectedTeam,
                onChange: (event) => setSelectedTeam(event.target.value),
              },
              ...teamOptions
            )
          ),
          createElement(
            "div",
            null,
            createElement("p", { className: "label" }, "Year"),
            createElement(
              "select",
              {
                value: selectedYear ?? "",
                onChange: (event) => setSelectedYear(Number(event.target.value)),
              },
              ...yearOptions
            )
          ),
          createElement(
            "div",
            { className: "team-summary" },
            createElement("p", { className: "summary-label" }, "Current selection"),
            createElement(
              "p",
              { className: "summary-value" },
              selectedTeam && selectedYear
                ? `${selectedTeam} — ${selectedYear}`
                : "Waiting for data…"
            )
          )
        ),
        createElement(
          "section",
          { className: "roster-footer" },
          createElement(
            "div",
            null,
            createElement("p", { className: "label" }, "Expected wins"),
            createElement("p", { className: "expected-wins" }, expectedWins)
          ),
          createElement(
            "div",
            null,
            createElement("p", { className: "label" }, "Actual wins"),
            createElement("p", { className: "expected-wins" }, actualWins)
          ),
          createElement(
            "button",
            { type: "button", className: "run-model", onClick: handleRunModel },
            "Run model"
          )
        ),
        !predictedWins &&
          createElement(
            "p",
            { className: "notice notice-strong" },
            createElement(
              "strong",
              null,
              "Expected wins will update after running the model on the current roster ordering."
            )
          ),
        createElement(
          "p",
          { className: "notice" },
          "Model runs on the current drag/swap order; this may differ from the original roster for this team/year."
        )
      ),
      // right column
      createElement(
        "main",
        { className: "roster-area" },
        createElement(
          "section",
          { className: "player-column" },
          !(teamValid && yearValid)
            ? createElement(
                "div",
                { className: "empty-state" },
                "Select a valid team and year."
              )
            : !roster
            ? createElement(
                "div",
                { className: "empty-state" },
                "No roster found for this team/year."
              )
            : playerTiles
        )
      )
    ),
    selectedPlayer
      ? createElement(
          "div",
          {
            className: "modal-backdrop",
            onClick: () => setSelectedPlayer(null),
          },
          createElement(
            "div",
            {
              className: "modal",
              onClick: (e) => e.stopPropagation(),
            },
            createElement(
              "div",
              { className: "modal-header" },
              createElement("h3", null, selectedPlayer.name || "Player details"),
              createElement(
                "button",
                { className: "close-btn", onClick: () => setSelectedPlayer(null) },
                "×"
              )
            ),
            createElement(
              "div",
              { className: "swap-panel" },
              createElement("p", { className: "label" }, "Swap player (name | year | team)"),
              createElement("input", {
                className: "swap-input",
                placeholder: "e.g., LeBron James | 2016 | CLE",
                value: swapSelection,
                onChange: (e) => setSwapSelection(e.target.value),
                list: "player-year-list",
              }),
              createElement(
                "datalist",
                { id: "player-year-list" },
                ...playerYearOptions.map((opt) => createElement("option", { key: opt, value: opt }))
              ),
              createElement(
                "button",
                {
                  className: "swap-btn",
                  onClick: handleSwap,
                  disabled: !swapSelection.includes("|"),
                },
                "Swap player"
              )
            ),
            createElement(
              "div",
              { className: "detail-grid" },
              ...Object.entries(selectedPlayer.stats || {})
                .filter(([k]) => !["age", "position"].includes(k.toLowerCase()))
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([key, val]) =>
                  createElement(
                    "div",
                    { key },
                    createElement("span", { className: "stat-label" }, key),
                    createElement("span", { className: "stat-value" }, formatStat(key, selectedPlayer.stats))
                  )
                )
            ),
          )
        )
      : null
  );
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(createElement(App));
