# NBA Team Win Prediction Dataset

A season-level dataset of NBA teams from 2016 onward, built for predicting regular season wins from roster composition. Each row represents one team-season, with aggregate team stats and individual statistics for each team's top 10 players by usage.

---

## Dataset Structure

Each row = one team and one season.

### Target Variable (Y)

| Column | Description |
|---|---|
| `reg_season_wins` | Regular season wins (prediction target) |
| `reg_losses` | Regular season losses |

---

### Team-Level Features

Aggregate metrics summarizing the roster as a whole.

| Column | Description |
|---|---|
| `season` | NBA season year |
| `team` | Team abbreviation |
| `team_avg_bpm` | Average Box Plus/Minus across qualified players |
| `team_avg_per` | Average Player Efficiency Rating across qualified players |
| `team_max_usg` | Highest individual usage rate on the roster |
| `team_players_qualified` | Number of players meeting a minutes threshold |
| `team_total_vorp` | Sum of Value Over Replacement Player across all players |
| `team_total_ws` | Sum of Win Shares across all players |
| `team_usg_gini` | Gini coefficient of usage rates — measures how concentrated ball-handling is |

---

### Player-Level Features (`p1_` through `p10_`)

Players are ranked 1–10 by usage rate (most-used player = `p1`). Each player slot has the same set of columns, prefixed by `p{n}_`.

#### Identity & Role

| Column | Description |
|---|---|
| `p{n}_name` | Player name |
| `p{n}_age` | Age during the season |
| `p{n}_position` | Position |
| `p{n}_games` | Games played |
| `p{n}_gamesStarted` | Games started |
| `p{n}_minutesPg` | Minutes per game |
| `p{n}_minutesPlayed` | Total minutes played |

#### Shooting

| Column | Description |
|---|---|
| `p{n}_fieldAttempts` | Field goal attempts |
| `p{n}_fieldGoals` | Field goals made |
| `p{n}_fieldPercent` | Field goal percentage |
| `p{n}_twoAttempts` | Two-point attempts |
| `p{n}_twoFg` | Two-point field goals made |
| `p{n}_twoPercent` | Two-point percentage |
| `p{n}_threeAttempts` | Three-point attempts |
| `p{n}_threeFg` | Three-point field goals made |
| `p{n}_threePercent` | Three-point percentage |
| `p{n}_threePAR` | Three-point attempt rate (3PA / FGA) |
| `p{n}_ft` | Free throws made |
| `p{n}_ftAttempts` | Free throw attempts |
| `p{n}_ftPercent` | Free throw percentage |
| `p{n}_ftr` | Free throw rate (FTA / FGA) |
| `p{n}_effectFgPercent` | Effective field goal % — adjusts for 3-pointers being worth more |
| `p{n}_tsPercent` | True shooting % — accounts for 2s, 3s, and free throws |

#### Production

| Column | Description |
|---|---|
| `p{n}_points` | Total points |
| `p{n}_assists` | Total assists |
| `p{n}_totalRb` | Total rebounds |
| `p{n}_offensiveRb` | Offensive rebounds |
| `p{n}_defensiveRb` | Defensive rebounds |
| `p{n}_steals` | Total steals |
| `p{n}_blocks` | Total blocks |
| `p{n}_turnovers` | Total turnovers |
| `p{n}_personalFouls` | Personal fouls |

#### Advanced Rate Stats

| Column | Description |
|---|---|
| `p{n}_usagePercent` | Usage rate — % of team plays used while on court |
| `p{n}_assistPercent` | % of teammate FGs assisted while on court |
| `p{n}_turnoverPercent` | Turnovers per 100 plays used |
| `p{n}_offensiveRBPercent` | % of available offensive rebounds grabbed |
| `p{n}_defensiveRBPercent` | % of available defensive rebounds grabbed |
| `p{n}_totalRBPercent` | % of available total rebounds grabbed |
| `p{n}_stealPercent` | % of opponent possessions ending in a steal |
| `p{n}_blockPercent` | % of opponent 2PA blocked while on court |

#### Value Metrics

| Column | Description |
|---|---|
| `p{n}_per` | Player Efficiency Rating |
| `p{n}_box` | Box Plus/Minus — overall on-court impact vs. average |
| `p{n}_offensiveBox` | Offensive Box Plus/Minus |
| `p{n}_defensiveBox` | Defensive Box Plus/Minus |
| `p{n}_vorp` | Value Over Replacement Player |
| `p{n}_winShares` | Total Win Shares |
| `p{n}_winSharesPer` | Win Shares per 48 minutes |
| `p{n}_offensiveWS` | Offensive Win Shares |
| `p{n}_defensiveWS` | Defensive Win Shares |

---