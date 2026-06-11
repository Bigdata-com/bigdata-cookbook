import { useState } from "react";

const COLORS = {
  bg: "#0a0e1a",
  card: "#111827",
  cardHover: "#1a2237",
  border: "#1e293b",
  borderActive: "#3b82f6",
  accent: "#ef4444",
  accentGlow: "rgba(239,68,68,0.15)",
  blue: "#3b82f6",
  blueGlow: "rgba(59,130,246,0.12)",
  amber: "#f59e0b",
  amberGlow: "rgba(245,158,11,0.1)",
  emerald: "#10b981",
  emeraldGlow: "rgba(16,185,129,0.1)",
  purple: "#8b5cf6",
  text: "#f1f5f9",
  textMuted: "#94a3b8",
  textDim: "#64748b",
  critical: "#ef4444",
  high: "#f59e0b",
  medium: "#3b82f6",
  low: "#10b981",
};

// ── Grounded Data — Updated Mar 18, 2026 08:04 UTC via Bigdata.com MCP ──
const GROUNDED_DATA = {
  energyMarkets: {
    brent:  { price: 102.10, d1: "-1.28%", d5: "-1.01%",  m1: "+51.44%", m3: "+71.08%", ytd: "+67.05%" },
    wti:    { price: 93.22,  d1: "-0.30%", d5: "-5.56%",  m1: "+46.78%", m3: "+66.64%", ytd: "+61.64%" },
    rbob:   { price: 3.11,   d1: "+0.84%", d5: "+2.17%",  m1: "+61.52%", m3: "+83.41%", ytd: "+80.97%" },
    heat:   { price: 3.77,   d1: "-0.93%", d5: "-6.16%",  m1: "+56.27%", m3: "+75.26%", ytd: "+76.57%" },
    natgas: { price: 2.94,   d1: "-3.07%", d5: "-6.10%",  m1: "-3.00%",  m3: "-26.94%", ytd: "-18.72%" },
    spread: "~$8.88",
    brentYearStart: "~$61.12",
    timestamp: "Mar 18, 2026 08:04 UTC",
    drivers: [
      { headline: "IRAQ-TURKEY CEYHAN DEAL — Oil Slides as Iraq Resumes Exports Bypassing Hormuz", detail: "Iraq and Kurdish authorities agreed to resume exports via Turkey's Ceyhan port, bypassing the Strait of Hormuz entirely — the biggest supply-side relief development since the war began. Brent fell to $101.09 (MT Newswires 07:02 UTC) before stabilizing at $102.10. WTI dropped to $93.05 before recovering to $93.22. ING: 'upstream production continues to decline as producers manage storage constraints; Brent has found a floor just above $100.' MSN (07:59 UTC): Brent -1.5%, WTI -2.3% early morning on the news. Hormuz flows still constrained — deal only covers Iraqi Kurdish exports.", attribution: "MT Newswires / MSN / ING, Mar 18 07:02–08:04 UTC" },
      { headline: "OCBC Revises Brent Profile: $100/bbl Through Mid-2026, ~$70 By Early 2027", detail: "OCBC strategists (Sim Moh Siong & Christopher Wong) revised their Brent profile higher, expecting ~$100/bbl through mid-2026 before easing toward $70 by early 2027. 'No clear path to de-escalation as US-Iran conflict enters week three; limited vessel movement keeps Hormuz effectively shut and oil flows at a near standstill.' Warns persistent shipping paralysis could turn temporary disruptions into lasting supply losses. Deutsche Bank (08:03 UTC): Brent above $100 but daily trading ranges narrowing; markets pricing longer disruption via 6-month Brent futures.", attribution: "OCBC / FXStreet / Deutsche Bank, Mar 18 07:18–08:03 UTC" },
      { headline: "FOMC Decision Imminent — Powell's Last Press Conference; Dot Plot Zero-Cuts Risk", detail: "FOMC announces rate decision today (Mar 18). 94% probability hold at 3.50-3.75%. Key signal: dot plot shift from 1 cut to 0 cuts in 2026 = 'significant hawkish shock' (Tong Yang Securities). FT academic poll: majority say $100 oil will markedly reduce US growth. Traders now expect only 1 cut (December). FXStreet: 'unambiguously bullish for dollar' if dot plot signals zero cuts. BofA: moderate oil shock 'bimodal' for policy. HSBC: 'geopolitical tensions reinforce USD's role as primary safe-haven.' Powell steps down May — this is a key press conference.", attribution: "FXStreet / Tong Yang / BofA / HSBC / FT, Mar 18 02:17–08:04 UTC" },
      { headline: "Iran Retaliates for Larijani Killing — Missiles on Israel + US Bases; Bushehr Struck", detail: "Iran launched drone/missile strikes after Israel killed security chief Ali Larijani. IRGC attacked Tel Aviv (2 killed near Tel Aviv per CNBC). US dropped 5,000-lb guided bombs on Iranian missile sites near Hormuz (CENTCOM/ABC). Hostile projectile struck Bushehr Nuclear Power Plant — IAEA confirmed no casualties, no facility damage. Iran also struck Saudi eastern energy region. US counterterrorism official resigned. Top US counterterrorism official resigned (Yahoo News 04:44 UTC). UAE/Kuwait further cut output; Saudi East-West pipeline at full capacity.", attribution: "CNBC / ABC / CENTCOM / IAEA / Yahoo News, Mar 18 02:56–04:44 UTC" },
      { headline: "Asian Surge: Nikkei +2.87%, KOSPI +5.04%; VIX Eases to 21.60; Gold $4,999", detail: "Asian markets led global rally on Iraq-Turkey deal + oil retreat + FOMC positioning. Nikkei 225: 55,239.40 (+2.87%, tearsheet 08:04 UTC) — highest since pre-war. KOSPI: 5,925.03 (+5.04%) — leading global gains, +40.60% YTD. NIFTY 50: 23,830.20 (+1.06%). Hang Seng 26,048.22 (+0.69%). VIX eased to 21.60 (-3.44%) from 22.37 at 06:03 UTC — risk-off sentiment fading. Gold: $4,998.60 (-0.19%). S&P 500: 6,716.09 (+0.25%). USD/JPY: 158.79. XLE (energy sector ETF): +30.87% YTD.", attribution: "Bigdata.com Market Tearsheet, Mar 18 08:04 UTC" },
    ],
    sources: [
      { headline: "Brent & WTI real-time prices and performance data", source: "Bigdata.com Market Tearsheet", ts: "2026-03-18T08:04:00", id: "MARKET_TEARSHEET", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-market-tearsheet" },
      { headline: "Oil slips as rising U.S. crude inventories offset attacks on UAE energy infrastructure", source: "CNBC", ts: "2026-03-18T02:56:14", id: "9160E0797F21ADEC61542FEA8E8F31B5", url: "https://www.cnbc.com/2026/03/18/oil-prices-brent-wti-uae-energy-attacks-us-crude-inventories-hormuz.html" },
      { headline: "Dozens of ships slip through the Strait of Hormuz as Iran's oil exports get through", source: "Associated Press", ts: "2026-03-18T05:15:11", id: "A188B3BA280492F48238358D64382C02", url: "https://apnews.com/article/ships-iran-oil-china-us-trump-hormuz-82a9acb473837f1bf7a821d0c3f95205" },
      { headline: "Iran launches retaliatory strikes on Israel and U.S. assets after security chief Larijani is killed", source: "CNBC", ts: "2026-03-18T02:56:15", id: "128522CF7DF3AED08264E1EF938123FB", url: "https://www.cnbc.com/2026/03/18/iran-strikes-us-israeli-targets-gulf-larijani-death.html" },
      { headline: "WTI weakens below $93.00 mark, eyes weekly low despite rising Middle East tensions", source: "FXStreet News", ts: "2026-03-18T03:59:50", id: "E3625855050A19B39EFF44B2327E499B", url: "https://app.bigdata.com/documents/E3625855050A19B39EFF44B2327E499B?cnum=4" },
      { headline: "Hormuz Reopening Looks Unlikely Without a Ceasefire in Iran War", source: "Insurance Journal", ts: "2026-03-18T05:45:15", id: "1F8753A77BE11A986D17ACF16DDF1B88", url: "https://www.insurancejournal.com/news/international/2026/03/18/862291.htm" },
      { headline: "Oil price surge from Iran war will hurt US growth and slow interest rate cuts, economists warn", source: "Financial Times", ts: "2026-03-18T01:32:33", id: "7B932B9964A29392710B47B7B1751D20", url: "https://www.ft.com/content/d4c3f857-eb50-47c5-a109-050a8c2f2346" },
      { headline: "EMEA Morning Briefing: Markets Await FOMC Decision", source: "Morningstar", ts: "2026-03-18T05:22:15", id: "A930ADB5F1BED6164BD4972D88A8B5BF", url: "https://www.morningstar.com/news/dow-jones/20260318337/emea-morning-briefing-markets-await-fomc-decision" },
      { headline: "Asian Stocks Rally as Oil Retreats, Fed in Spotlight", source: "US News & World Report", ts: "2026-03-18T02:35:58", id: "241192CDF35F0073571816A0715791E1", url: "https://money.usnews.com/investing/news/articles/2026-03-17/asian-stocks-rally-as-oil-retreats-fed-in-spotlight" },
      { headline: "Brent crude rises to three-and-a-half-year high as Iran widens strikes on energy targets", source: "MSN", ts: "2026-03-18T00:18:44", id: "4CC2833F227C11995EF0B6DD524E495A", url: "https://www.msn.com/en-us/money/markets/brent-crude-rises-to-three-and-a-half-year-high-as-iran-widens-strikes-on-energy-targets/ar-AA1YRCFW" },
      { headline: "Oil falls as Iraq strikes Ceyhan deal but Hormuz constraints remain; Brent floor ~$100", source: "MSN", ts: "2026-03-18T07:59:50", id: "909B46263E66FB5E5E588D9759B077D4", url: "https://www.msn.com/en-us/money/markets/oil-falls-as-iraq-strikes-oil-exports-deal-but-hormuz-constraints-remain/ar-AA1YS29d" },
      { headline: "Brent: Elevated conflict risk supports higher path — OCBC $100 through mid-2026", source: "FXStreet News", ts: "2026-03-18T07:18:28", id: "F66D4E10A35E4D465B7D300387988B7A", url: "https://app.bigdata.com/documents/F66D4E10A35E4D465B7D300387988B7A?cnum=1" },
      { headline: "Oil: Volatility eases as daily trading ranges narrow — Deutsche Bank", source: "FXStreet News", ts: "2026-03-18T08:03:59", id: "8507EA085795E2607F135975F836EBD7", url: "https://app.bigdata.com/documents/8507EA085795E2607F135975F836EBD7?cnum=1" },
    ],
  },
  goldmanAnalysis: {
    q2Forecast: { value: "$101 Brent", detail: "GS expects Brent >$100 avg March, ~$85 in April (3-week disruption). UBS: $120 end-March if flows don't improve, $150 in Q2. Brent $103.41 (-0.01%) at 00:04 UTC Wed, WTI $95.12 (+1.73%). RBA raised to 4.10% — first G10 hike amid oil shock. FOMC decision Mar 18 — dot plot zero-cuts risk (Tong Yang: 'significant hawkish shock'). BofA: Iran war shock 'bimodal' for policy. GS global growth cut to 2.6%." },
    upside: { value: "$150 peak UBS", detail: "UBS (Patricot): $120 end-March if no improvement, $150 Q2 prolonged scenario. GS adverse: $130. GS very adverse: $150. Iran FM: $200 warning. Rystad worst-case: ME crude could fall to 6M bpd (70% drop). BCA: shock 'more globally disruptive than 2022.' BlackRock CEO Fink: below $50 if Iran neutralized." },
    q4Forecast: { value: "5,400 bear GS", detail: "GS: S&P 500 year-end 7,600 (21x). Moderate shock: 6,300 (19x). Bear case: 5,400 (-19%) at $150 oil. GS head hedge fund coverage: 'market underestimates potential downside tails' (CNBC 13:32 UTC). S&P 500 6,719 (+0.30%); VIX 22.91 (-2.55%) at 16:03 UTC — calm could be false; FOMC decision tomorrow. Recession probability 25%. GS recommends solar, cybersecurity, defensive sectors." },
    riskPremium: "$13–18/bbl",
    riskPremiumPct: "~25%",
    sources: [
      { headline: "Iran war unlikely to trigger global supply chain crisis, Goldman Sachs says", source: "FOX Business", ts: "2026-03-16T16:48:23", id: "573F4C24AC6419E53868167120292411", url: "https://www.foxbusiness.com/economy/iran-war-unlikely-trigger-global-supply-chain-crisis-goldman-sachs-says" },
      { headline: "Markets May Be Underpricing Iran Risks, Bank of America Warns", source: "Yahoo! Finance", ts: "2026-03-16T15:58:28", id: "A34AE5E8FFE65D7BA2EEB97583B4DCF0", url: "https://finance.yahoo.com/news/markets-may-underpricing-iran-risks-145912133.html" },
      { headline: "Goldman Sachs flags 3 investment opportunities as Iran war tests bull rally", source: "AOL.com / Goldman Sachs", ts: "2026-03-16T15:11:54", id: "74D8D08227353D2A47FEB89B90DCEFC3", url: "https://www.aol.com/articles/goldman-sachs-flags-3-investment-142244322.html" },
      { headline: "Could The Iran War Trigger An S&P 500 Bear Market?", source: "Benzinga", ts: "2026-03-16T14:30:05", id: "2100B1A12808C83AF4BD20DB853F4BD5", url: "https://www.benzinga.com/node/51271411" },
      { headline: "Oil prices hover near $100 per barrel 3 weeks into Iran war", source: "Yahoo! Finance", ts: "2026-03-16T14:20:58", id: "DFBA41A9B60D17CE8E6C33989261A4F3", url: "https://finance.yahoo.com/video/oil-prices-hover-near-100-140726228.html" },
      { headline: "Strategists Stay Upbeat on US Stocks Despite Iran War Risks", source: "Bloomberg Law", ts: "2026-03-16T10:54:12", id: "7549D416FEB869790C9E76C2788D6994", url: "https://news.bloomberglaw.com/international-trade/strategists-stay-upbeat-on-us-stocks-despite-iran-war-risks" },
      { headline: "Goldman Sachs thinks stock market underestimates Iran war risk after Monday's bounce", source: "CNBC", ts: "2026-03-17T13:32:50", id: "BE2ABF3FEC46D76040F692E380343438", url: "https://www.cnbc.com/2026/03/17/goldman-sachs-thinks-stock-market-underestimates-iran-war-risk-after-mondays-bounce.html" },
      { headline: "S&P 500, VIX, Dow, Nasdaq real-time index levels and performance", source: "Bigdata.com Market Tearsheet", ts: "2026-03-18T08:04:00", id: "MARKET_TEARSHEET", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-market-tearsheet" },
      { headline: "The Stocks Goldman Sachs Thinks You Should Own as Iran War Stretches Into Third Week", source: "Yahoo! Finance", ts: "2026-03-16T20:28:49", id: "13B86E877C2F09AF998A0EA9542C4483", url: "https://finance.yahoo.com/news/stocks-goldman-sachs-thinks-own-191840907.html" },
    ],
  },
  hormuz: {
    globalOilTransitPct: "~20%",
    status: "Day 19 (08:04 UTC) — Iraq-Turkey Ceyhan Deal; Brent Holds ~$102",
    statusDetail: "DAY 19 08:04 UTC — IRAQ-TURKEY CEYHAN DEAL: Iraq and Kurdish authorities agreed to resume oil exports via Turkey's Ceyhan port, bypassing Hormuz entirely — largest supply-side relief since war began. Brent $102.10 (-1.28%), WTI $93.22 (-0.30%) at 08:04 UTC. ING: 'Brent has found a floor just above $100.' OCBC (07:18 UTC): Brent ~$100 through mid-2026, easing to ~$70 by early 2027 — 'no clear path to de-escalation.' DEUTSCHE BANK (08:03 UTC): daily ranges narrowing; 6-month Brent futures pricing longer disruption. IRAN RETALIATES FOR LARIJANI KILLING: IRGC drone/missile attack on Tel Aviv (2 killed); hostile projectile struck Bushehr Nuclear Plant (no damage per IAEA); Iran struck Saudi eastern energy region. US DROPS 5,000-LB GUIDED BOMBS on Iranian missile sites near Hormuz (CENTCOM). FOMC DECISION TODAY — 94% hold; dot plot zero-cuts = 'significant hawkish shock' (Tong Yang). ASIAN RALLY: Nikkei 55,239 +2.87%, KOSPI 5,925 +5.04%, VIX 21.60 (-3.44%). 'Hormuz Reopening Looks Unlikely Without a Ceasefire' (Insurance Journal).",
    carriersSuspended: "All 5 majors (Maersk, MSC, CMA CGM, Hapag-Lloyd, COSCO) suspended Hormuz; CMA CGM restarted Gulf shipping but skirts Hormuz entirely; Maersk also paused Red Sea/Bab el-Mandeb transits; Hapag-Lloyd ship 'Source Blessing' hit by debris near Hormuz; MSC 59% of recorded container diversions (project44); Hormuz risk premiums up 300% (Yahoo Finance); 18+ commercial ships struck since war began",
    rerouteVia: "Cape of Good Hope (+15–20 days)",
    trafficDrop: "Down >96%",
    accessLevel: "CONDITIONAL",
    shipsStruck: 18,
    alternativeRoutes: [
      { name: "Iraq-Turkey Ceyhan", status: "ACTIVE", capacityMbd: "~0.4 mbd" },
      { name: "Saudi Yanbu (Red Sea)", status: "ACTIVE", capacityMbd: "~3 mbd" },
      { name: "Jask (Iran bypass)", status: "LIMITED", capacityMbd: "~0.3 mbd" },
    ],
    sources: [
      { headline: "Mojtaba Khamenei said to reject ceasefire talks, demand US/Israel 'brought to their knees'", source: "Yahoo! News", ts: "2026-03-17T14:26:22", id: "0A025F86A69B3A0413433973E755E462", url: "https://www.yahoo.com/news/articles/mojtaba-khamenei-said-reject-ceasefire-141133172.html" },
      { headline: "Iran's new supreme leader rejects calls for ceasefire with US", source: "MSN", ts: "2026-03-17T12:31:51", id: "D411684BD7531DFAF5FB8F8C2FF594AC", url: "https://www.msn.com/en-gb/news/world/iran-s-new-supreme-leader-rejects-calls-for-ceasefire-with-us/ar-AA1YOKOw" },
      { headline: "Dubai airport halts flights again as UAE shuts airspace", source: "Forbes.com", ts: "2026-03-17T14:55:51", id: "C44116752B001DD7ABD6D664691E1EE6", url: "https://www.forbesindia.com/article/news/dubai-airport-halts-flights-again-as-uae-shuts-airspace/2992257/1" },
      { headline: "Trump Must Battle 'Hydra Holdout' — PGIM: mine deployment extends to months", source: "Benzinga", ts: "2026-03-17T13:47:50", id: "4E11937FF69C6C76805EFDDEA60752E3", url: "https://www.benzinga.com/node/51296734" },
      { headline: "Iran targets UAE energy infrastructure as gas field set ablaze, tanker struck near Hormuz", source: "CNBC", ts: "2026-03-17T10:29:08", id: "C9F548DC44EB9B42A40542E2977D3B70", url: "https://www.cnbc.com/2026/03/17/iran-war-uae-energy-gas-field-oil-fujairah-strait-of-hormuz.html" },
      { headline: "Senior Iran Official Says Trade Through the Strait of Hormuz Will Never Be The Same", source: "AOL.com", ts: "2026-03-17T16:57:18", id: "A9FAEC176C38B8B6F71C4DD4298464DD", url: "https://www.aol.com/news/senior-iran-official-says-trade-133728698.html" },
      { headline: "Israel Has 'Won' War With Iran, Foreign Minister Says, but Goals Remain Unmet", source: "US News & World Report", ts: "2026-03-17T17:03:16", id: "9D9509B5CB7897584C627EC095B40CAC", url: "https://www.usnews.com/news/world/articles/2026-03-17/israel-has-won-war-with-iran-foreign-minister-says-but-goals-remain-unmet" },
      { headline: "Oil prices jump as Iran warns Strait of Hormuz 'cannot be the same'", source: "CNN", ts: "2026-03-17T10:21:17", id: "44517309D7293F8B764A3E6A5402B2C9", url: "https://www.cnn.com/2026/03/17/business/oil-prices-strait-iran-attacks-intl" },
      { headline: "Trump warns NATO over Hormuz Strait — Why Europe and allies rejected his demand", source: "ZeeNews", ts: "2026-03-17T00:00:56", id: "FB0F37B883A489538697A94B77582B41", url: "https://zeenews.india.com/world/trump-warns-nato-over-hormuz-strait-why-europe-and-allies-rejected-his-demand-3027483.html" },
      { headline: "Near-term energy supply shock unavoidable — feedback loops in focus", source: "Fund Library", ts: "2026-03-17T00:01:21", id: "F8152124C73AB33B9F78CA2D49CCE49A", url: "https://www.fundlibrary.com/Articles/Detail/near-term-energy-supply-shock-unavoidable/2151" },
      { headline: "Oil falls as Iraq strikes oil exports deal but Hormuz constraints remain", source: "MSN", ts: "2026-03-18T07:59:50", id: "909B46263E66FB5E5E588D9759B077D4", url: "https://www.msn.com/en-us/money/markets/oil-falls-as-iraq-strikes-oil-exports-deal-but-hormuz-constraints-remain/ar-AA1YS29d" },
      { headline: "Brent: Elevated conflict risk supports higher path — OCBC targets $100 through mid-2026", source: "FXStreet News", ts: "2026-03-18T07:18:28", id: "F66D4E10A35E4D465B7D300387988B7A", url: "https://app.bigdata.com/documents/F66D4E10A35E4D465B7D300387988B7A?cnum=1" },
      { headline: "Oil: Volatility eases as supply headlines shift — Deutsche Bank; Iraq-Turkey deal cited", source: "FXStreet News", ts: "2026-03-18T08:03:59", id: "8507EA085795E2607F135975F836EBD7", url: "https://app.bigdata.com/documents/8507EA085795E2607F135975F836EBD7?cnum=1" },
    ],
  },
  dualChokepoint: {
    description: "First simultaneous compromise of Hormuz + Suez/Bab el-Mandeb in modern history — IEA: 'largest supply disruption in history of the global oil market'",
    seaborneCrudeAffected: "~31–38% of global seaborne crude; Rystad: 12M+ bpd taken offline; Julius Baer: ~10M bpd shut-ins (~10% of global supply); Rystad worst-case: ME crude could fall to 6M bpd",
    houthiStatus: "ESCALATION: Houthis 'fingers on the trigger' — Fars News warns activation could shut Bab el-Mandeb Strait. Bab el-Mandeb handles 8.8M bpd (~10–12% of seaborne oil). Economic Times: 'double chokepoint' scenario forming. Maersk paused Red Sea/Bab el-Mandeb transits. Atlantic Council: disrupting Red Sea 'more impactful and far riskier' in 2026. Firstpost: activating Houthis as maritime deterrent is 'logical step' for Tehran. Axios: Houthis are 'new core force' in Axis of Resistance.",
    qatarWarning: "Goldman: $145+ if flows at current levels; GS: largest oil supply shock on record; Julius Baer: oil above $100 with inflation risks rising; ING: 'only way to see oil trade lower is getting oil flowing through Hormuz'; Capital Economics: 3-month war + damage could hit $150; Iran warns $200/bbl; CNBC: prolonged standoff threatens America's generic drug prescriptions; RBC: conflict could last 'well into the spring'; Rystad worst-case: ME crude production falls to 6M bpd (70% drop)",
    sources: [
      { headline: "Oil prices in for more shock if Houthis close Bab al-Mandeb", source: "The Economic Times", ts: "2026-03-16T12:18:40", id: "A561FE3ABE18D5C2AA904B8E2B07497B", url: "https://economictimes.indiatimes.com/news/international/us/oil-prices-in-for-more-shock-what-could-happen-if-the-houthis-of-yemen-close-the-bab-al-mandab-strait/articleshow/129609085.cms" },
      { headline: "After Hormuz, another chokepoint? Red Sea on edge as Houthis threaten", source: "MSN", ts: "2026-03-15T06:53:21", id: "AA9568F163766A4C61C3806A0986441D" },
      { headline: "After Strait of Hormuz, is Bab el-Mandeb the next target?", source: "ZeeNews", ts: "2026-03-16T11:20:58", id: "DC778E287942125436870F114FF7463A", url: "https://zeenews.india.com/world/after-strait-of-hormuz-is-bab-el-mandeb-the-next-target-if-houthis-join-iran-war-3027400.html" },
      { headline: "How Houthis backing Iran could trigger a double chokepoint crisis", source: "Firstpost", ts: "2026-03-16T07:58:18", id: "D7CB4A1C82E2D8B6F7F6FE0DE5A1B982", url: "https://www.firstpost.com/opinion/iran-houthis-oil-chokepoints-13989986.html" },
      { headline: "Will the Houthis join the Iran war?", source: "Atlantic Council", ts: "2026-03-10T16:14:08", id: "DFBDEE60FCA505D45594BF49B1A282E4", url: "https://www.atlanticcouncil.org/blogs/menasource/will-the-houthis-join-the-iran-war/" },
      { headline: "Axis of Resistance mobilizes second front for Iran", source: "MSN", ts: "2026-03-16T02:28:47", id: "B4A0525E37011ED948CC6F0222D45334" },
    ],
  },
  countries: [
    { name: "Japan", flag: "🇯🇵", meOilDep: "90–95%", hormuzDep: "~65–70%", reserves: "254 days; BEGAN releasing 80M bbl — 15 days private sector first, 1 month state later; 7th time since 1970s; not planning escort mission; PM Takaichi to visit Washington; refineries cutting production; naphtha shortages hitting petrochemicals (FT)", risk: 5, riskLabel: "Critical", color: COLORS.critical, cbPolicy: "HOLD" },
    { name: "South Korea", flag: "🇰🇷", meOilDep: "70–71%", hormuzDep: "65%", reserves: "208 days; $68B market-stabilisation programme activated; fuel price caps imposed (first since 1997); LIFTS coal cap, boosts nuclear to ~80%; naphtha shortages hitting petrochemicals — SK imports 60% from Gulf (FT); KOSPI +1.14% Mon (+31.69% 3M); reviewing Trump's Hormuz request", risk: 5, riskLabel: "Critical", color: COLORS.critical, cbPolicy: "HOLD" },
    { name: "Taiwan", flag: "🇹🇼", meOilDep: "60%+", hormuzDep: "~60%", reserves: "Enough for March (govt); semiconductor industry vulnerable — 25% of world helium supply at risk; fossils dominate (<10% renewables)", risk: 5, riskLabel: "Critical", color: COLORS.critical, cbPolicy: "HOLD" },
    { name: "India", flag: "🇮🇳", meOilDep: "50–55%", hormuzDep: "~50%", reserves: "~20–60 days; Jaishankar: 'no blanket arrangement' with Iran for Indian-flagged ships; 4+ tankers crossed safely; India asking for 23 more; 25% of gas supply affected by force majeure; ramping Russian crude with US waiver; rupee pinned to lifetime low; NIFTY +1.11% Mon", risk: 4, riskLabel: "High", color: COLORS.high, cbPolicy: "HOLD" },
    { name: "China", flag: "🇨🇳", meOilDep: "40–54% via Hormuz", hormuzDep: "~50%", reserves: "~100–220 days (1.1–1.4B bbl); Natixis: 220 days; Iran shipped 11M+ bbl to China since war began; importing via Jask bypass; Iran may let tankers pass if oil sold in yuan; retail sales + industrial output beat forecast; Trump may delay Beijing visit", risk: 4, riskLabel: "High", color: COLORS.high, cbPolicy: "CUT" },
    { name: "Qatar", flag: "🇶🇦", meOilDep: "N/A (LNG exp.)", hormuzDep: "100% (LNG halted)", reserves: "LNG exporter — force majeure declared; warns $150/bbl", risk: 5, riskLabel: "Critical", color: COLORS.critical, cbPolicy: "N/A" },
    { name: "UAE", flag: "🇦🇪", meOilDep: "N/A (producer)", hormuzDep: "100% via strait", reserves: "Under attack — Fujairah Port RESUMED Sun; ADNOC Ruwais refinery shut; Dubai airport temporarily CLOSED by Iranian drone; IRGC fired on US base in UAE Mon; Iran warns 3 major ports to evacuate; UAE indices extending losing streak", risk: 5, riskLabel: "Critical", color: COLORS.critical, cbPolicy: "N/A" },
    { name: "Saudi Arabia", flag: "🇸🇦", meOilDep: "N/A (producer)", hormuzDep: "89% of exports", reserves: "Producer — Aramco CEO: 'catastrophic consequences'; record 5.9M bpd from Yanbu; Houthi 'Hour Zero' threat via Bab el-Mandeb; Tadawul +0.44%; exploring channels to reduce tensions", risk: 4, riskLabel: "High", color: COLORS.high, cbPolicy: "N/A" },
    { name: "Germany/EU", flag: "🇪🇺", meOilDep: "18%", hormuzDep: "Low direct", reserves: "90 days IEA; EU's Kallas floats Black Sea model for Hormuz; France holds back warships; Germany: no NATO role in strait; European FMs meeting in Brussels; DAX +0.97% Mon; STOXX 600 +0.84%; UK mortgage rates jump as lenders pull products", risk: 2, riskLabel: "Medium", color: COLORS.medium, cbPolicy: "HOLD" },
    { name: "United States", flag: "🇺🇸", meOilDep: "2%", hormuzDep: "Minimal", reserves: "SPR ~400M bbl → releasing 172M bbl; gas prices up ~24% since war; S&P 500 6,699 (+1.01%) best day in 5 wks; VIX 23.51 (-13.53%); Gold $5,017.50; FOMC meets Mar 17–18 TODAY — 92-94% hold at 3.50–3.75%; Senate gives Trump broad authority; 47% traders see zero cuts 2026 (FT); Bessent: 'fine with some ships transiting'; Trump warns NATO over Hormuz; delays China summit", risk: 1, riskLabel: "Low", color: COLORS.low, cbPolicy: "HOLD" },
  ],
  chinaDeep: {
    iranCrudeImportsMbd: "1.38",
    iranShareOfImports: "~13.4%",
    meShareOfSeaborne: { value: "~45–50%", detail: "Kpler/Natixis/RTE Ireland" },
    meSeaborneMbd: "~5.0–5.4 mbd via Hormuz",
    sprBillionBbl: "~1.2–1.4",
    sprCoverDays: "~200–222",
    russianPivot: "ONLY Iranian tankers guaranteed Hormuz passage — ~90% of Iranian oil flowing to China (Foreign Policy Mar 17). China importing via Jask port + pipeline bypassing Hormuz (Epoch Times). Iran shipped 11M+ bbl to China since war began (CNBC). Iran allowing oil passage if sold in yuan. Natixis: SPR covers 222 days (1.2B bbl / 5.4M bpd). Washington Post (Mar 17): China has ~1.2B bbl storage — 'cover several months of imports.' Banned refined fuel exports (NDRC). Retail sales + industrial output beat forecast. Goehring & Rozencwajg: adversaries 'choking oil through Hormuz' — validating China's reserves strategy.",
    actions: "FM Wang Yi: ceasefire 'highest priority.' Trump may delay Beijing visit (FT) — pressuring China on Hormuz. US-China economic chiefs met in Paris — Iran oil on agenda. Bessent announced 30-day Russian oil waiver. Washington Post: prolonged war 'could hand China the commodity it prizes most' — discounted oil at scale. CNPC: reserves 'certainly exceed' 90 days. Natixis: China's oil deficit 1–1.4M bpd — artisanal refiners most affected. Societe Generale: ~1.5B bbl SPR — ~200 days. Power of Siberia 2 pipeline project fast-tracked in 15th Five-Year Plan.",
    sources: [
      { headline: "A prolonged war in Iran could hand China the commodity it prizes most", source: "Washington Post", ts: "2026-03-17T20:00:50", id: "936CE665212B6E7A5B55805C75B96F1E", url: "https://www.washingtonpost.com/opinions/2026/03/17/iran-china-trump/" },
      { headline: "China's Hormuz Problem — only Iranian tankers pass; all heading to China", source: "Foreign Policy Magazine", ts: "2026-03-17T21:07:49", id: "8E2D4A785E11FF9757E2FB02502EC4D2", url: "https://foreignpolicy.com/2026/03/17/iran-middle-east-war-china-strait-hormuz-oil-energy-exports/" },
      { headline: "Will the war in Iran break the Chinese model? Xi Jinping's dark scenarios", source: "Natixis", ts: "2026-03-17T12:29:39", id: "B17AF9B5A3BAF3F6B2F31A0F4FC25309", url: "https://research.bluematrix.com/docs/pdf/3de41bc7-bfd2-406d-a320-bef7107a3f1c/0a910bbc-9ed5-45b3-9146-e1fff0cb4594" },
      { headline: "China Continues Importing Iranian Oil Through 'Backdoor' Route", source: "Epoch Times", ts: "2026-03-14T23:11:05", id: "A31C0A85C60AA84D17C384D94D039B44", url: "https://www.theepochtimes.com/world/china-continues-importing-iranian-oil-through-backdoor-route-bypassing-strait-of-hormuz-5998848" },
      { headline: "Geopolitical Tensions in Iran and Implications for China — SPR covers 222 days", source: "Natixis", ts: "2026-03-16T02:22:51", id: "EDC6FB464FE37DE3BE5A935B938B4A2A" },
      { headline: "China to lean on Russian oil as Iran crisis chokes supply", source: "Financial Times", ts: "2026-03-04T04:00:47", id: "E75B47258385E1140B334497D65D45A5", url: "https://www.ft.com/content/114997aa-7d7c-4d85-b696-bc5123ade6cb" },
      { headline: "US Strikes On Iran Tighten Pressure On China's Energy Links", source: "Benzinga", ts: "2026-03-06T17:02:19", id: "AD6AE50C69E2E752436BFC90CFA1A115", url: "https://www.benzinga.com/node/51103188" },
      { headline: "China macro overview: GDP, CPI, indices, CNY, economic calendar", source: "Bigdata.com Country Tearsheet — CN", ts: "2026-03-18T08:04:00", id: "COUNTRY_TEARSHEET_CN", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-country-tearsheet" },
    ],
  },
  timeline: [
    { date: "Jun 2025", event: "Israel + US strike Iran nuclear facilities (Operation Midnight Hammer)", oilImpact: "Brent spiked from ~$65 to low $80s, quickly retraced on ceasefire" },
    { date: "Jan 2026", event: "Iran exports drop to ~1.5 mbd; US-Iran Geneva talks (3rd round) begin", oilImpact: "IEA warns oversupply; WTI ~$63; Brent ~$68" },
    { date: "Feb 25, 2026", event: "US sanctions Iran shadow fleet & arms networks; intelligence shows imminent Iran attack", oilImpact: "Secondary sanctions risk flagged; prices stable near $66" },
    { date: "Feb 28, 2026", event: "Operation Epic Fury: US + Israel joint strikes; Khamenei killed", oilImpact: "Brent from $66 → $78 by Mar 2; oil +28% on the week" },
    { date: "Mar 1, 2026", event: "Iran retaliates — missiles/drones hit US bases across ME; Hapag-Lloyd halts Hormuz transits", oilImpact: "All major carriers follow; Brent crosses $80" },
    { date: "Mar 2, 2026", event: "Iran IRGC declares Hormuz officially closed; WTI biggest 1-day rally since Mar 2022", oilImpact: "WTI +7%; Brent crosses $82; GS $18/bbl risk premium" },
    { date: "Mar 3–4, 2026", event: "9 vessels attacked in Gulf; COSCO halts all Hormuz bookings; Iran strikes UAE/Oman/Bahrain/Kuwait", oilImpact: "GS raises Q2 Brent to $76 avg (+$10); Iraq may cut 3+ mbd" },
    { date: "Mar 5, 2026", event: "Israel moves to 'next phase'; Iran strikes oil tanker in Persian Gulf; LNG disrupted; VIX >25", oilImpact: "WTI $81 (+6%); Brent $85 — highest since Jul 2024; KOSPI worst day on record" },
    { date: "Mar 6, 2026", event: "Trump demands Iran 'UNCONDITIONAL SURRENDER'; Russia providing intel to Iran; Qatar warns $150/bbl; NFP shocks at -92K", oilImpact: "WTI surges to $91.66 (+12%); Brent $93.72 — highest since 2023; GS warns $100 imminent" },
    { date: "Mar 7, 2026", event: "War enters Week 2; Kuwait declares force majeure; earthquake hits Tehran; ~600 ships stranded", oilImpact: "WTI $90.90; Brent $92.69 — +36%/+28% weekly (biggest since 1983); S&P -2% wk; VIX 29.5" },
    { date: "Mar 8, 2026", event: "Israel strikes Tehran oil depots — first attack on Iran's oil infrastructure; Goldman confirms Hormuz flows down 90%", oilImpact: "Oil crosses $100 — first time since 2022; Goldman: $150/bbl if war unresolved" },
    { date: "Mar 9, 2026", event: "HISTORIC WHIPSAW: Oil spiked to $119.50/$119.48 intraday then crashed as G7 reserve signal + Trump told CBS war 'very complete'. Nikkei -7%; KOSPI -8.2% (circuit breaker). Mojtaba Khamenei named Supreme Leader.", oilImpact: "WTI $94.77 (Mon close); Brent $98.96; intraday high $119.50; Goldman: $150 if Hormuz closed >30 days" },
    { date: "Mar 10, 2026", event: "Oil crashed 12% at settlement on Trump 'war ending soon.' Iran began laying mines (CNN). US destroyed 16 minelayers + 10 boats (CENTCOM/Trump). Oil whipsawed $80–$90+ on mine headlines. G7/IEA emergency meetings.", oilImpact: "WTI $83.45 (-11.9%); Brent $87.80 (-11%); wild swings on mine news; still +24%/+35% from pre-war $67" },
    { date: "Mar 11, 2026", event: "Day 12: IEA CONFIRMED record 400M bbl release. IRGC 'heaviest operation' targeting Israel + US bases in 5 countries. Feb CPI inline 2.4%. Dow -289. Iran FM warns $200/bbl. Gulf output cut ~6.7M bpd. Pentagon: war cost $11.3B first week.", oilImpact: "WTI $87.25 (+4.6% settle); Brent $91.98 (+4.8%); oil rebounds despite IEA release" },
    { date: "Mar 12, 2026", event: "Day 13: IEA: 'LARGEST SUPPLY DISRUPTION IN HISTORY' — Gulf cut 10M+ bpd. Mojtaba Khamenei: 'Hormuz must remain closed.' Brent topped $101 again. 6+ more vessels struck — tankers ablaze at Iraq Basra port. Iraq halted ALL oil port ops. Goldman raises Q4 to $71; warns $145+ peak.", oilImpact: "Brent $100.46 (+9.2%); WTI $95.75 (+9.7%); +49%/+42% from pre-war $67" },
    { date: "Mar 13, 2026", event: "Day 14: Intense airstrikes hit Tehran early Fri (AP). 6 US airmen confirmed dead in Iraq crash. Goldman Fri note: Brent avg >$100 March — forecast revised 20% higher. US issued 30-day waiver for Russian oil purchases. France & Italy enter talks with Iran for ship passage. CMA CGM restarts Gulf shipping skirts Hormuz. RBC: conflict could last 'well into the spring.'", oilImpact: "Brent ~$103.14 Fri settle (+2.67%); WTI ~$98.71 (+3.11%); ~42% above pre-war $73" },
    { date: "Mar 14, 2026", event: "Day 15: US STRIKES KHARG ISLAND — Iran's main oil export hub. IRAN DRONE STRIKES FUJAIRAH PORT. ADNOC Ruwais refinery shut. IRGC: US interests in UAE 'legitimate targets.' 2,500 Marines deploying. HOUTHI 'HOUR ZERO' declared. Trump REJECTS ceasefire. S&P 500 3rd weekly loss. Dow -0.3%, Nasdaq -0.9% for week.", oilImpact: "WTI settled $99.31 (+3.74%); Brent settled $103.14 (+3.43%); highest weekly close since 2022" },
    { date: "Mar 15, 2026", event: "Day 16 — War enters THIRD WEEK. Pentagon: war up to 6 weeks. Fujairah Port RESUMED. Iran FM on Face the Nation: 'Strait open to non-enemies.' IEA Sunday: 400M+ bbl to flow. IRGC struck 3 more US bases (Harir, Ali al Salem, Arifjan). Russia supplying Shahed drones to Iran (Zelenskyy). Baghdad airport base hit TWICE.", oilImpact: "Sunday futures: WTI opened $100.22, Brent opened $106.11 — both surging past $100 again" },
    { date: "Mar 16, 2026", event: "Day 17 CLOSE: Multiple tankers navigate Hormuz safely over weekend — oil slides sharply. India negotiating passage for 6+ more vessels; other countries using back channels to Iran. FIRST NON-IRANIAN AIS TRANSIT: Pakistani oil tanker crossed Hormuz with transponder activated (Marine Traffic). Treasury Sec Bessent: US 'fine with some Iranian, Indian and Chinese ships going through Hormuz for now.' Goldman Sachs formally states Iran war 'unlikely to trigger global supply chain crisis' — oil shock only; GDP hit -0.3%, inflation +0.5-0.6pp. BCA Research: shock 'more globally disruptive than 2022.' IDF ground ops in Lebanon + strikes on Tehran/Shiraz/Tabriz/Hamadan. Iran publishes full demand list — permanent ceasefire, Netanyahu handover, sanctions lifted, reparations. Senate gives Trump broad war authority. FOMC begins Mar 17-18.", oilImpact: "Brent $101.16 (-1.92%), WTI $94.46 (-4.31%) at 18:02 UTC. Gold $4,993. VIX 24.01 (-11.70%). S&P 500 6,698 (+1.00%). Oil +65% YTD from pre-war ~$61." },
    { date: "Mar 17, 2026", event: "Day 18 EOD: IRAN OFFICIAL QALIBAF 'HORMUZ CAN NEVER BE THE SAME' — parliament speaker declares oil trade through strait permanently altered; oil surged +3% to $103.02+ on statement (AOL 16:57 UTC). ISRAEL FM SAAR: 'We have already won' war but goals remain unmet; IDF continues strikes on Tehran, Lebanon, nuclear/missile sites. IRAN ATTACKS US EMBASSY IN BAGHDAD — most intense assault since war began. ISRAEL KILLS IRAN'S BASIJ CHIEF Gholamreza Soleimani. DIESEL HITS $5/GALLON nationally — highest since 2022 (CNBC). FOMC Day 1 ends; dot plot could signal ZERO cuts in 2026. RBA HIKED RATES overnight — first G10 central bank to raise amid oil shock. Araghchi denies US contact; disputed channel reduces ceasefire probability. CENTCOM: 15,000+ Iranian targets struck; 18 ships struck since war began. Heating Oil surged to $3.80 (+6.06%). KOSPI 5,640; Wall St best gain in 5 weeks.", oilImpact: "Brent $103.02 (+2.80%), WTI $95.09 (+1.70%) at 18:02 UTC. Gold $5,004.70 (+0.05%). S&P 500 6,724.70 (+0.38%). VIX 22.27 (-5.27%). Energy sector XLE +1.45%. +68.55% Brent YTD from pre-war ~$61.12." },
    { date: "Mar 18, 2026", event: "Day 19: IRAQ-TURKEY CEYHAN DEAL — Iraq and Kurdish authorities agree to resume oil exports via Turkey's Ceyhan port, bypassing Strait of Hormuz entirely (MT Newswires 07:02 UTC). FOMC holds 3.50-3.75% (94%); dot plot zero-cuts signal hawkish risk. ISRAEL KILLS ALI LARIJANI (Iran's security chief) in overnight strike; Iran retaliates — drone/missile strike on Tel Aviv (2 killed), Bushehr Nuclear Plant struck (no damage per IAEA), Saudi eastern energy region hit. Dozens of ships slipping through Hormuz (AP 05:15 UTC) — first notable transit improvement since war began. JPMORGAN: Hormuz transit 'INCREASINGLY CONDITIONAL.' ING: 'Brent has found a floor just above $100.' OCBC: ~$100/bbl through mid-2026, easing to ~$70 by early 2027. CITI warns: 4–6 week disruption = 11–16M bpd removed, Brent $110–$120. DEUTSCHE BANK: 6-month Brent futures pricing longer disruption. VIX 21.60; KOSPI +5.04% to 5,925; Nikkei +2.87%.", oilImpact: "Brent $102.10 (-1.28%), WTI $93.22 (-0.30%), RBOB $3.11 (+0.84%), Heat $3.77 (-0.93%), NatGas $2.94 (-3.07%) at 08:04 UTC from Bigdata.com Market Tearsheet. Gold $4,998.60. S&P 500 6,716 +0.25%. VIX 21.60. +67.05% Brent YTD from pre-war ~$61.12." },
  ],
  countrySources: [
    { headline: "Japan releases emergency oil reserves in 80M bbl drawdown", source: "ANI", ts: "2026-03-16T04:00:00", id: "F319A25F9C64ABD17EE5BB270BA67F23" },
    { headline: "Japan to 'act first' to release oil reserves amid Hormuz crisis", source: "Financial Times", ts: "2026-03-11T13:32:42", id: "559E549659218CE15FB7F7332521CB35", url: "https://www.ft.com/content/ffc35d71-2596-4472-9332-1fcaa123eeae" },
    { headline: "Gulf oil shock deepens crisis for Asia's petrochemicals industry", source: "Financial Times", ts: "2026-03-13T06:38:17", id: "9821E5E98DC08011D63947521ED1C990", url: "https://www.ft.com/content/cb21c436-3bcc-4211-8f92-8041e5a1c698" },
    { headline: "Asia's Energy Triage Amid the Iran War", source: "The Diplomat", ts: "2026-03-11T18:40:50", id: "5947110DBD1E8CA03D75CDF72DEACFC2", url: "https://thediplomat.com/2026/03/asias-energy-triage-amid-the-iran-war/" },
    { headline: "Oil price spike likely to keep rates on hold — deepen Fed divisions", source: "Yahoo! Finance", ts: "2026-03-16T11:57:25", id: "9EE37F8DCA44185E80273956B2F2122D", url: "https://nz.finance.yahoo.com/news/oil-price-spike-likely-to-keep-rates-on-hold-but-deepen-divisions-among-fed-officials-this-week-090015969.html" },
    { headline: "Iran hits Gulf neighbors — keeps stranglehold on oil shipping", source: "New York Post", ts: "2026-03-16T10:45:01", id: "E9B09E49286905BEB0AAFF407F5C3F7F", url: "https://nypost.com/2026/03/16/world-news/iran-hits-gulf-neighbors-and-keeps-stranglehold-on-oil-shipping-as-concerns-rise-of-energy-crisis/" },
    { headline: "UK mortgage rates jump as lenders pull products amid Iran war", source: "MSN", ts: "2026-03-16T11:33:50", id: "9ADA39BA0C5AAA364F66684DE3413D9E" },
      { headline: "Country equity ETFs, major indexes, and currency performance data", source: "Bigdata.com Market Tearsheet", ts: "2026-03-18T08:04:00", id: "MARKET_TEARSHEET", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-market-tearsheet" },
      { headline: "Japan macro overview: GDP, CPI, indices, JPY, economic calendar", source: "Bigdata.com Country Tearsheet — JP", ts: "2026-03-18T08:04:00", id: "COUNTRY_TEARSHEET_JP", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-country-tearsheet" },
      { headline: "China macro overview: GDP, CPI, indices, CNY, economic calendar", source: "Bigdata.com Country Tearsheet — CN", ts: "2026-03-18T08:04:00", id: "COUNTRY_TEARSHEET_CN", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-country-tearsheet" },
      { headline: "India macro overview: GDP, CPI, NIFTY, INR, economic calendar", source: "Bigdata.com Country Tearsheet — IN", ts: "2026-03-18T08:04:00", id: "COUNTRY_TEARSHEET_IN", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-country-tearsheet" },
      { headline: "US macro overview: GDP, CPI, yields, DXY, FOMC, economic calendar", source: "Bigdata.com Country Tearsheet — US", ts: "2026-03-18T08:04:00", id: "COUNTRY_TEARSHEET_US", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-country-tearsheet" },
    { headline: "Why the timing of the war against Iran is exceptionally brutal for India", source: "The New Indian Express", ts: "2026-03-16T15:47:23", id: "CA27A6380FBA4C66EE8FE67EAE6FB757", url: "https://www.newindianexpress.com/web-only/2026/Mar/16/why-the-timing-of-the-war-against-iran-is-exceptionally-brutal-for-india" },
    { headline: "Europeans seek clarity about Trump's Iran war aims before agreeing to his warship demands", source: "MSN", ts: "2026-03-16T15:56:06", id: "352BD07B16E4BA062838045D639F8657", url: "https://www.msn.com/en-ca/money/topstories/europeans-seek-clarity-about-trump-s-iran-war-aims-before-agreeing-to-his-warship-demands/ar-AA1YIWJJ" },
  ],
  timelineSources: [
    { headline: "Iran Demands Permanent Ceasefire And War Reparations", source: "International Business Times", ts: "2026-03-16T11:32:12", id: "44AA8B5162C6D8E754884C0D31E1A205", url: "https://www.ibtimes.com/iran-demands-permanent-ceasefire-war-reparations-signalling-firm-stance-against-us-israeli-forces-3799257" },
    { headline: "Oil prices ease as US says it is fine with some ships going through Hormuz", source: "MSN", ts: "2026-03-16T16:01:42", id: "2A8BB96D016A5D52F13329032A2EE807", url: "https://www.msn.com/en-gb/news/world/oil-prices-ease-as-us-says-it-is-fine-with-some-ships-going-through-strait-of-hormuz/ar-AA1YKxYx" },
    { headline: "Senior Iran Official Says Trade Through the Strait of Hormuz Will Never Be The Same", source: "AOL.com", ts: "2026-03-17T16:57:18", id: "A9FAEC176C38B8B6F71C4DD4298464DD", url: "https://www.aol.com/news/senior-iran-official-says-trade-133728698.html" },
    { headline: "Mojtaba Khamenei rejects ceasefire, demands US and Israel 'brought to their knees'", source: "Yahoo! News", ts: "2026-03-17T14:26:22", id: "0A025F86A69B3A0413433973E755E462", url: "https://www.yahoo.com/news/articles/mojtaba-khamenei-said-reject-ceasefire-141133172.html" },
    { headline: "Diesel prices surge to $5 per gallon, highest since 2022, as Iran war disrupts global oil supplies", source: "CNBC", ts: "2026-03-17T13:48:35", id: "BAC59E87D32F23D13E9F71E19E71B37A", url: "https://www.cnbc.com/2026/03/17/diesel-gas-oil-price-iran-war-hormuz.html" },
    { headline: "Oil Settles Higher on Supply Threats — WTI +2.9%, Brent above $100 4th session; Larijani killed", source: "Rigzone", ts: "2026-03-18T00:01:24", id: "E58F3B18E5659318885E32D1B3F21CE3", url: "https://www.rigzone.com/news/wire/oil_settles_higher_on_supply_threats-17-mar-2026-183231-article/" },
    { headline: "Dozens of ships slip through the Strait of Hormuz as Iran's oil exports get through", source: "Associated Press", ts: "2026-03-18T05:15:11", id: "A188B3BA280492F48238358D64382C02", url: "https://apnews.com/article/ships-iran-oil-china-us-trump-hormuz-82a9acb473837f1bf7a821d0c3f95205" },
    { headline: "Iran launches retaliatory strikes on Israel and U.S. assets after security chief Larijani is killed", source: "CNBC", ts: "2026-03-18T02:56:15", id: "128522CF7DF3AED08264E1EF938123FB", url: "https://www.cnbc.com/2026/03/18/iran-strikes-us-israeli-targets-gulf-larijani-death.html" },
    { headline: "Oil falls as Iraq strikes oil exports deal but Hormuz constraints remain; ING floor ~$100", source: "MSN", ts: "2026-03-18T07:59:50", id: "909B46263E66FB5E5E588D9759B077D4", url: "https://www.msn.com/en-us/money/markets/oil-falls-as-iraq-strikes-oil-exports-deal-but-hormuz-constraints-remain/ar-AA1YS29d" },
    { headline: "Brent: Elevated conflict risk — OCBC revises profile to $100 through mid-2026", source: "FXStreet News", ts: "2026-03-18T07:18:28", id: "F66D4E10A35E4D465B7D300387988B7A", url: "https://app.bigdata.com/documents/F66D4E10A35E4D465B7D300387988B7A?cnum=1" },
    { headline: "Oil: Volatility eases as daily trading ranges narrow — Deutsche Bank", source: "FXStreet News", ts: "2026-03-18T08:03:59", id: "8507EA085795E2607F135975F836EBD7", url: "https://app.bigdata.com/documents/8507EA085795E2607F135975F836EBD7?cnum=1" },
    { headline: "Oil prices and performance data", source: "Bigdata.com Market Tearsheet", ts: "2026-03-18T08:04:00", id: "MARKET_TEARSHEET", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-market-tearsheet" },
  ],
  mindmapNodes: {
    root: { id: "root", label: "Iran Escalation — Day 19 08:04 UTC", type: "root" },
    military: [
      { id: "m1", label: "IDF Strikes + Ground Ops; First Tanker Transit", detail: "IDF launched new wave of strikes on Tehran, Shiraz, Tabriz, Hamadan. IDF begins 'targeted' ground operations in southern Lebanon (AOL). IDF planning 3+ more weeks. Trump: Iran's missiles 'down to a low number.' IRGC fired on US base in UAE; claims 80%+ destruction at 3 US bases. Dubai airport temporarily CLOSED. Over 2,000 dead (majority in Iran).", confidence: 0.98, timeHorizon: "immediate" },
      { id: "m2", label: "MINE WARFARE — Persistent Threat", detail: "CRS 19-page report: reopening Hormuz could take months. 2 of 3 US mine-hunting ships in ASIA not Middle East. Goldman: Hormuz flows at 600K bpd vs 19M+ normal. ING: 'even if flows resume, ramp-up takes time.' Hormuz risk premiums up 300%. Oil prices held firm despite 400M bbl IEA release.", confidence: 0.96, timeHorizon: "immediate" },
      { id: "m3", label: "22+ Ships Struck — Attacks Widening", detail: "AP: 12M+ bpd taken offline (Rystad). Dubai airport hit by drone. Iraq halted ALL oil port ops. Only 77 ships since war vs 100/day pre-war. CMA CGM restarts Gulf but skirts Hormuz. MSC 59% of diversions. Julius Baer: ~10M bpd shut-ins (~10% of global supply). Rystad worst-case: ME crude could fall to 6M bpd.", confidence: 0.97, timeHorizon: "immediate" },
      { id: "m4", label: "Larijani Killed; Hormuz 'Increasingly Conditional' — JPMorgan", detail: "Israel killed Ali Larijani, Iran's security chief, in overnight strikes (Westpac Mar 18) — major decapitation strike on Iran's security establishment. JPMorgan (Kaneva): Hormuz transit 'increasingly conditional' — Iran selectively allowing some vessels based on affiliation; ships hugging Iran coast. UAE and Kuwait further cut output. Saudi Yanbu at full capacity via East-West bypass. IEA: 8M bpd cut (7.5% global supply). Mine clearing could take months (CRS report). No ceasefire channel open — Khamenei still demands US/Israel 'brought to their knees.'", confidence: 0.98, timeHorizon: "medium" },
      { id: "m5", label: "Houthi 'HOUR ZERO' — Bab el-Mandeb Threat", detail: "Houthis 'fingers on the trigger.' Bab el-Mandeb handles 8.8M bpd (~10–12% seaborne oil). Economic Times: 'double chokepoint' forming. Maersk paused Red Sea transits. Atlantic Council: disrupting Red Sea 'more impactful and far riskier' in 2026. Axios: Houthis are 'new core force' in Axis of Resistance.", confidence: 0.75, timeHorizon: "immediate" },
    ],
    energy: [
      { id: "e1", label: "IEA: 'LARGEST DISRUPTION IN HISTORY'", detail: "IEA confirms 'largest supply disruption in history.' Goldman: Hormuz flows at 600K bpd vs 19M+ normal. Rystad: 12M+ bpd taken offline. Julius Baer: ~10M bpd shut-ins (~10% global supply). Rystad worst-case: ME crude could fall to 6M bpd (70% drop). ING: 'only way to see oil trade lower is oil flowing through Hormuz.'", confidence: 0.99, timeHorizon: "immediate" },
      { id: "e2", label: "Oil $102.76; Heat $3.80 +6.06%; Diesel $5/Gallon", detail: "Brent $102.76 (+2.54%), WTI $94.97 (+1.57%) at 16:03 UTC. HEATING OIL $3.80 (+6.06%) — highest level this cycle reflecting distillate tightness. DIESEL hits $5/gallon nationally — highest since 2022 (CNBC 13:48). RBOB $3.11 (+5.45%); NatGas $3.05 (+0.99%). RABOBANK: Iran targeting 'upstream fields not just flows.' DANSKE BANK: prices high 'even if Hormuz resumes.' Dubai airport closed again. Prabhudas Lilladher: ~10M bpd supply gap by April 1. Oil +68.13% YTD. Gold $5,000.80.", confidence: 0.98, timeHorizon: "immediate" },
      { id: "e3", label: "GS: Oil Shock But Not Supply Chain Crisis", detail: "Goldman: Iran war driving oil shock but supply chains should hold — modest GDP hit (~0.3%), little risk of pandemic-style chaos. Inflation +0.5–0.6%. GS base: Brent $98 avg Mar, $85 Apr. UBS: Brent $90 June, $85 YE. GS adverse: $130 peak. GS very adverse: $150. Recession probability 25%.", confidence: 0.96, timeHorizon: "immediate" },
      { id: "e4", label: "IEA 400M + Japan 80M = Insufficient", detail: "Japan BEGAN releasing 80M bbl (7th time since 1970s). IEA: 400M total. US releasing 172M bbl. JPMorgan: reserves cover only 7.5% of shock. Goldman: release lowers prices ~$7/bbl but 'can't close gap.' Oil rallied THROUGH the release. Birol: 'opening Hormuz vital for stable flows.' Bessent: duration is key.", confidence: 0.97, timeHorizon: "immediate" },
      { id: "e5", label: "Supply Chain Contagion Widening", detail: "CNBC: prolonged standoff threatens America's generic drug prescriptions. FT: naphtha shortages hitting Japan/SK petrochemicals — SK imports 60% from Gulf. 30% of global fertilizers, 48% of traded sulfur transit region. India: 25% of gas hit by force majeure. Oil execs warn crunch worsening (WSJ). Piper Sandler Energy Conference this week.", confidence: 0.92, timeHorizon: "structural" },
    ],
    trade: [
      { id: "t1", label: "Dual Chokepoint — Maersk Pauses Red Sea", detail: "Hormuz + Bab el-Mandeb both threatened — ~31–38% of seaborne crude at risk. Houthis 'fingers on the trigger.' Maersk paused Red Sea transits. Rystad: 12M+ bpd offline. Only 77 ships since war vs 100/day pre-war. CMA CGM restarted Gulf but skirts Hormuz. Hormuz = world's most expensive waterway — 300% risk premium surge.", confidence: 0.96, timeHorizon: "immediate" },
      { id: "t2", label: "Coalition Friction — Allies Refusing", detail: "Trump demands allies help secure Hormuz — Japan + Australia say no escort ships. EU's Kallas floats Black Sea model. France holds back warships. Germany: no NATO role in strait. UK 'discussing options.' Iran's formal demand list raises the bar for any negotiation. South Korea lifting coal cap, boosting nuclear.", confidence: 0.90, timeHorizon: "immediate" },
      { id: "t3", label: "Iraq-Turkey Ceyhan Deal — First Hormuz Bypass Route Open", detail: "Iraq and Kurdish authorities agreed to resume crude exports via Turkey's Ceyhan port (Mar 18, 07:02 UTC) — first alternative export route activated since war began. Brent fell on the news. ING: 'Brent has found a floor just above $100; upstream production declining as producers manage storage.' This deal covers only Iraqi Kurdish exports — Saudi/Kuwait/UAE still bottlenecked. Mine-clearing still needed: CRS report 19 pages, could take months. 2 of 3 US mine-hunting ships in ASIA. OCBC: even with deal, Brent ~$100 through mid-2026.", confidence: 0.92, timeHorizon: "immediate" },
      { id: "t4", label: "Asia 14M bpd Not Arriving", detail: "China, India, Japan, SK collectively demand ~14M bpd of ME crude — not arriving. Japan releasing 80M bbl. Asian refineries forced to cut runs by 30% (PVM). Iran may let tankers pass if oil sold in yuan. CNBC: prolonged standoff threatens US generic drug prescriptions. FT: naphtha shortages hitting petrochemicals.", confidence: 0.92, timeHorizon: "immediate" },
    ],
    diplomatic: [
      { id: "d1", label: "Khamenei REJECTS Ceasefire — 'Brought to Knees'", detail: "Supreme Leader Mojtaba Khamenei rejected BOTH intermediary de-escalation proposals — demands US/Israel 'brought to their knees and accept defeat' (Reuters 14:26 UTC). This is a formal ruling by Iran's top authority, not just FM Araghchi's statements. Iran also attacked US embassy in Baghdad (most intense assault). Araghchi: 'war must end so enemies never again think of repeating attacks.' Zero open channels confirmed. Iran quietly pursuing separate deals with Iraq, Pakistan for oil passage — tactical maneuvering not strategic ceasefire.", confidence: 0.99, timeHorizon: "immediate" },
      { id: "d2", label: "EU Kallas Floats Black Sea Model", detail: "EU's Kallas proposes Black Sea-style initiative for Hormuz. France holds back warships. Germany: no NATO role. Japan + Australia say no escort ships. UK 'discussing options.' European FMs gathering in Brussels. Trump demands 7+ countries send ships. Trump may delay Beijing visit.", confidence: 0.82, timeHorizon: "immediate" },
      { id: "d3", label: "Araghchi DENIES Contact — US-Iran Channel Disputed", detail: "Iran FM Araghchi publicly denied direct contact with US envoy Witkoff (Benzinga/MSN 07:15 UTC): 'My last contact was prior to decision to kill diplomacy with another illegal military attack on Iran. Any claim geared solely to mislead oil traders.' US official: 'he was lying and initiated the contact.' Disputed channel = near-zero formal ceasefire probability near-term. Germany, Spain, Italy formally rebuffed Trump's warship demand. IMO chief: escorts won't guarantee safe passage (FT 05:00). Zero countries formally committed.", confidence: 0.91, timeHorizon: "immediate" },
      { id: "d4", label: "IEA/G7 Reserves — Physically Insufficient", detail: "Japan BEGAN 80M bbl release. IEA: 400M accessible. US 172M. JPMorgan: covers only 7.5%. Goldman: lowers prices ~$7/bbl but 'can't close gap.' Birol: 'opening Hormuz vital.' Oil rallied THROUGH the release. Bessent: 'duration of conflict key.' Oil execs warn crunch worsening.", confidence: 0.97, timeHorizon: "immediate" },
      { id: "d5", label: "IDF Ground Ops in Lebanon — War Expanding", detail: "IDF begins 'targeted' ground operations in southern Lebanon (AOL Mar 16). Conflict expanding beyond Iran to Hezbollah front. Multiple ME fronts simultaneously. Zelenskyy: countries requesting Ukraine's anti-drone expertise. Russia supplying Shahed drones to Iran. Baghdad airport base hit twice.", confidence: 0.88, timeHorizon: "immediate" },
    ],
    financial: [
      { id: "f1", label: "Iraq-Turkey Deal + FOMC — Oil $102/$93; VIX 21.60; Ceyhan Bypass", detail: "IRAQ-TURKEY CEYHAN DEAL (07:02 UTC): Iraq resumes exports via Turkey's Ceyhan port, bypassing Hormuz — first major supply-side bypass since war. Brent $102.10 (-1.28%), WTI $93.22 (-0.30%) at 08:04 UTC. ING: 'Brent floor just above $100.' OCBC: Brent ~$100 through mid-2026. FOMC announces rate decision today — 94% hold; dot plot zero-cuts = 'significant hawkish shock' (Tong Yang). FT poll: majority say $100 oil markedly reduces US growth. VIX 21.60 (-3.44%). S&P 500 6,716.09 (+0.25%). Gold $4,998.60 (-0.19%). Nikkei +2.87%, KOSPI +5.04% leading Asian rally. Citi: 4-6 week disruption = Brent $110–$120. Moody's Zandi: recession 'difficult to avoid' if oil elevated.", confidence: 0.99, timeHorizon: "immediate" },
      { id: "f2", label: "Fed Paralyzed — Rate Cut Repricing", detail: "47% see zero cuts in 2026 — up from 5% a month ago (FT). TD Securities: hold through Q3, first cut September. EY-Parthenon: only one 0.25% cut in Dec. Deutsche Bank: dot plot signals one cut. MUFG: $100 oil adds ~0.8pp to inflation; $150 scenario pushes above 4%. Wells Fargo: 'worst nightmare.' GS year-end PCE 2.9%.", confidence: 0.97, timeHorizon: "immediate" },
      { id: "f3", label: "Goldman: Recession Probability 25%", detail: "GS recession probability 25% (up 5pp). GS: S&P 500 could fall to 6,300 if growth weakens. Morgan Stanley: oil doubling could shave 1.5% off US GDP. GS: oil shock but not supply chain crisis — modest GDP hit. Kalshi: recession odds ~31%. But GS/MS/JPM: US stocks case 'remains intact' (Bloomberg Law).", confidence: 0.94, timeHorizon: "medium" },
      { id: "f4", label: "Asian Emergency Measures — Stocks Surging as Oil Retreats", detail: "Japan releasing 80M bbl. South Korea: price caps first since 1997; KOSPI 5,917 +4.90% Wed (tearsheet) — regional leader; naphtha shortages. Nikkei 55,151 +2.70% Wed. India: NIFTY 23,794 +0.90%; rupee at lifetime low. Indonesia: JKSE -17.81% YTD — worst-performing major market. Asian stocks rallying as oil retreats from peak and dozens of ships slip through Hormuz. Asian refineries contemplating 30% run cuts (PVM); Power of Siberia 2 fast-tracked.", confidence: 0.97, timeHorizon: "immediate" },
    ],
  },
};

// ── Utility Components ──

function Badge({ children, color = COLORS.accent, glow = COLORS.accentGlow }) {
  return (
    <span style={{ display: "inline-block", padding: "2px 10px", borderRadius: 999, fontSize: 13, fontWeight: 700, letterSpacing: 0.5, color, background: glow, border: `1px solid ${color}33`, textTransform: "uppercase", fontFamily: "'Inter', -apple-system, sans-serif" }}>
      {children}
    </span>
  );
}

function Metric({ label, value, sub, color = COLORS.text }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ fontSize: 13, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 1, fontWeight: 600, fontFamily: "'Inter', -apple-system, sans-serif" }}>{label}</span>
      <span style={{ fontSize: 24, fontWeight: 800, color, fontFamily: "'JetBrains Mono', monospace", letterSpacing: -0.5 }}>{value}</span>
      {sub && <span style={{ fontSize: 13, color: COLORS.textMuted, fontFamily: "'Inter', -apple-system, sans-serif" }}>{sub}</span>}
    </div>
  );
}

function SourceTag({ sources }) {
  const [expanded, setExpanded] = useState(false);
  if (!sources || sources.length === 0) return null;
  const fmtDate = (ts) => { try { const d = new Date(ts); return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }); } catch { return ts; } };
  return (
    <div style={{ marginTop: 8, borderTop: `1px solid ${COLORS.border}`, paddingTop: 6 }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{ fontSize: 12, color: COLORS.textDim, cursor: "pointer", display: "flex", alignItems: "center", gap: 4, fontFamily: "'Inter', -apple-system, sans-serif", userSelect: "none" }}
      >
        <span style={{ fontSize: 10, transition: "transform 0.15s", transform: expanded ? "rotate(90deg)" : "rotate(0deg)", display: "inline-block" }}>&#9654;</span>
        <span style={{ fontStyle: "italic" }}>{sources.length} sources cited</span>
      </div>
      {expanded && (
        <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 6 }}>
          {sources.map((s, i) => (
            <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start", padding: "6px 8px", background: `${COLORS.border}22`, borderRadius: 4, borderLeft: `2px solid ${COLORS.blue}33` }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: COLORS.text, fontFamily: "'Inter', -apple-system, sans-serif", lineHeight: 1.3, marginBottom: 2, overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>{s.headline}</div>
                <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 11, color: COLORS.blue, fontWeight: 600, fontFamily: "'Inter', -apple-system, sans-serif" }}>{s.source}</span>
                  <span style={{ fontSize: 11, color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace" }}>{fmtDate(s.ts)}</span>
                  <span style={{ fontSize: 10, color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace", opacity: 0.6 }}>{s.id}</span>
                </div>
              </div>
              {s.url && (
                <a href={s.url} target="_blank" rel="noopener" style={{ fontSize: 11, color: COLORS.blue, textDecoration: "none", fontFamily: "'Inter', -apple-system, sans-serif", flexShrink: 0, padding: "2px 6px", border: `1px solid ${COLORS.blue}33`, borderRadius: 3 }}>↗</a>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CardContainer({ children, title, badge, accent = COLORS.blue, style = {} }) {
  return (
    <div style={{ background: COLORS.card, borderRadius: 12, border: `1px solid ${COLORS.border}`, padding: "20px 24px", ...style }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{ width: 4, height: 20, borderRadius: 2, background: accent, flexShrink: 0 }} />
        <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: COLORS.text, textTransform: "uppercase", letterSpacing: 1.2, fontFamily: "'Inter', -apple-system, sans-serif" }}>{title}</h3>
        {badge && <Badge color={accent} glow={`${accent}18`}>{badge}</Badge>}
      </div>
      {children}
    </div>
  );
}

// ── Panel Components ──

function EnergyMarketsPanel() {
  const d = GROUNDED_DATA.energyMarkets;
  const pctColor = (s) => s.startsWith("-") ? COLORS.accent : COLORS.emerald;
  const rows = [
    { name: "Brent Crude", data: d.brent },
    { name: "WTI Crude", data: d.wti },
    { name: "Gasoline RBOB", data: d.rbob },
    { name: "Heating Oil", data: d.heat },
    { name: "Natural Gas", data: d.natgas },
  ];
  return (
    <CardContainer title="Energy Markets" badge={`Brent $${d.brent.price}`} accent={COLORS.accent}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <span style={{ fontSize: 24, fontWeight: 800, color: COLORS.amber, fontFamily: "'JetBrains Mono', monospace", letterSpacing: -0.5 }}>${d.brent.price}</span>
          <span style={{ fontSize: 14, fontWeight: 700, color: pctColor(d.brent.d1), fontFamily: "'JetBrains Mono', monospace" }}>
            {d.brent.d1.startsWith("-") ? "▼" : "▲"} {d.brent.d1} today
          </span>
        </div>
        <span style={{ fontSize: 12, color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace" }}>{d.timestamp}</span>
      </div>
      <div style={{ overflowX: "auto", marginBottom: 12 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead>
            <tr style={{ borderBottom: `2px solid ${COLORS.border}` }}>
              {["Instrument", "Price", "1D", "5D", "1M", "3M", "YTD"].map((h) => (
                <th key={h} style={{ padding: "6px 8px", textAlign: h === "Instrument" ? "left" : "right", color: COLORS.textDim, fontWeight: 700, fontSize: 12, textTransform: "uppercase", letterSpacing: 1, fontFamily: "'Inter', -apple-system, sans-serif" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.name} style={{ borderBottom: `1px solid ${COLORS.border}`, background: i % 2 === 0 ? "transparent" : `${COLORS.border}22` }}>
                <td style={{ padding: "6px 8px", fontWeight: 700, color: COLORS.text, fontFamily: "'Inter', -apple-system, sans-serif" }}>{r.name}</td>
                <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "'JetBrains Mono', monospace", color: COLORS.text }}>
                  ${r.data.price}
                </td>
                {[r.data.d1, r.data.d5, r.data.m1, r.data.m3, r.data.ytd].map((val, j) => (
                  <td key={j} style={{ padding: "6px 8px", textAlign: "right", fontFamily: "'JetBrains Mono', monospace", color: pctColor(val), fontSize: 13 }}>{val}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: "flex", gap: 16, marginBottom: 12 }}>
        <Metric label="WTI-Brent Spread" value={d.spread} sub="WTI discount to Brent" color={COLORS.textMuted} />
        <Metric label="Brent Jan 1" value={d.brentYearStart} sub="Start of 2026 reference" color={COLORS.textMuted} />
      </div>
      {d.drivers && d.drivers.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8, fontFamily: "'Inter', -apple-system, sans-serif" }}>Key Drivers & News</div>
          {d.drivers.map((drv, i) => (
            <div key={i} style={{ marginBottom: 8, paddingLeft: 10, borderLeft: `2px solid ${COLORS.border}` }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text, fontFamily: "'Inter', -apple-system, sans-serif" }}>{drv.headline}</div>
              <div style={{ fontSize: 13, color: COLORS.textMuted, lineHeight: 1.4, fontFamily: "'Inter', -apple-system, sans-serif" }}>
                {drv.detail} — <span style={{ fontStyle: "italic", color: COLORS.textDim }}>{drv.attribution}</span>
              </div>
            </div>
          ))}
        </div>
      )}
      <SourceTag sources={d.sources} />
    </CardContainer>
  );
}

const ACCESS_LEVEL_COLOR = {
  CLOSED: COLORS.accent,
  CONDITIONAL: COLORS.amber,
  SELECTIVE: COLORS.amber,
  OPEN: COLORS.emerald,
};

function HormuzStatusPanel() {
  const d = GROUNDED_DATA.hormuz;
  const dd = GROUNDED_DATA.dualChokepoint;
  const accessColor = ACCESS_LEVEL_COLOR[d.accessLevel] ?? COLORS.textDim;
  return (
      <CardContainer title="Maritime Chokepoints" badge="Day 19 — Iraq-Turkey Ceyhan Deal" accent={COLORS.accent}>
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
          <span style={{ fontSize: 26 }}>🚢</span>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
              <span style={{ fontSize: 16, fontWeight: 800, color: COLORS.accent, fontFamily: "'Inter', -apple-system, sans-serif" }}>Strait of Hormuz — {d.status}</span>
              <Badge color={accessColor} glow={`${accessColor}18`}>{d.accessLevel}</Badge>
            </div>
            <div style={{ fontSize: 13, color: COLORS.textMuted, fontFamily: "'Inter', -apple-system, sans-serif" }}>{d.statusDetail}</div>
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 10 }}>
          <div style={{ background: `${COLORS.accent}11`, borderRadius: 6, padding: 10, border: `1px solid ${COLORS.accent}22` }}>
            <Metric label="Traffic Status" value={d.trafficDrop} sub="77 ships total since war vs 100/day pre-war; GS: 3% of normal" color={COLORS.accent} />
          </div>
          <div style={{ background: `${COLORS.accent}11`, borderRadius: 6, padding: 10, border: `1px solid ${COLORS.accent}22` }}>
            <Metric label="Offline Output" value="12M+ bpd" sub="Rystad Energy via AP; Julius Baer: ~10M bpd shut-ins (~10% global)" color={COLORS.accent} />
          </div>
          <div style={{ background: `${COLORS.accent}11`, borderRadius: 6, padding: 10, border: `1px solid ${COLORS.accent}22` }}>
            <Metric label="Ships Struck" value={String(d.shipsStruck)} sub="since war began" color={COLORS.accent} />
          </div>
        </div>
        <div style={{ fontSize: 13, color: COLORS.textMuted, lineHeight: 1.6, marginBottom: 10, fontFamily: "'Inter', -apple-system, sans-serif" }}>
          Carries <span style={{ color: COLORS.text, fontWeight: 700 }}>{d.globalOilTransitPct}</span> of global oil. {d.carriersSuspended}. Rerouting via {d.rerouteVia}. 44,000+ companies affected. US struck Kharg Island. Fujairah Port RESUMED Sun. European FMs meeting in Brussels. 2 of 3 US mine-hunting ships in ASIA.
        </div>
      </div>
      <div style={{ background: `${COLORS.accent}11`, borderRadius: 8, padding: 12, border: `1px solid ${COLORS.accent}22`, marginBottom: 10 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.accent, marginBottom: 4, fontFamily: "'Inter', -apple-system, sans-serif" }}>⚠ HOUTHI &apos;HOUR ZERO&apos; — Dual Chokepoint Crisis + NATO Coalition Friction</div>
        <div style={{ fontSize: 13, color: COLORS.textMuted, lineHeight: 1.5, fontFamily: "'Inter', -apple-system, sans-serif" }}>
          {dd.description}. ~<span style={{ color: COLORS.text, fontWeight: 600 }}>{dd.seaborneCrudeAffected}</span> trade compromised. {dd.houthiStatus} {dd.qatarWarning}.
        </div>
      </div>
      <div style={{ marginBottom: 10 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6, fontFamily: "'Inter', -apple-system, sans-serif" }}>Alternative Routes</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {d.alternativeRoutes.map((r) => (
            <span key={r.name} style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 10px", borderRadius: 999, fontSize: 12, fontWeight: 600, fontFamily: "'Inter', -apple-system, sans-serif", background: r.status === "ACTIVE" ? COLORS.amberGlow : `${COLORS.textDim}18`, border: `1px solid ${r.status === "ACTIVE" ? COLORS.amber : COLORS.textDim}44`, color: r.status === "ACTIVE" ? COLORS.amber : COLORS.textDim }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: r.status === "ACTIVE" ? COLORS.amber : COLORS.textDim, display: "inline-block", flexShrink: 0 }} />
              {r.name} — {r.capacityMbd}
            </span>
          ))}
        </div>
      </div>
      <SourceTag sources={[...d.sources, ...dd.sources]} />
    </CardContainer>
  );
}

function CountryExposurePanel() {
  const [sortBy, setSortBy] = useState("risk");
  const countries = [...GROUNDED_DATA.countries].sort((a, b) => sortBy === "risk" ? b.risk - a.risk : a.name.localeCompare(b.name));
  return (
    <CardContainer title="Country Exposure Matrix" badge="10 Economies" accent={COLORS.amber} style={{ gridColumn: "1 / -1" }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {["risk", "name"].map((s) => (
          <button key={s} onClick={() => setSortBy(s)} style={{ padding: "4px 12px", borderRadius: 6, fontSize: 13, fontWeight: 700, cursor: "pointer", border: `1px solid ${sortBy === s ? COLORS.amber : COLORS.border}`, background: sortBy === s ? COLORS.amberGlow : "transparent", color: sortBy === s ? COLORS.amber : COLORS.textDim, textTransform: "uppercase", letterSpacing: 0.8, fontFamily: "'Inter', -apple-system, sans-serif" }}>
            Sort by {s}
          </button>
        ))}
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead>
            <tr style={{ borderBottom: `2px solid ${COLORS.border}` }}>
              {["Country", "ME Oil Dep.", "Hormuz Dep.", "SPR / Reserves", "Risk Level", "CB Policy"].map((h) => (
                <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: COLORS.textDim, fontWeight: 700, fontSize: 12, textTransform: "uppercase", letterSpacing: 1.2, fontFamily: "'Inter', -apple-system, sans-serif" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {countries.map((c, i) => (
              <tr key={c.name} style={{ borderBottom: `1px solid ${COLORS.border}`, background: i % 2 === 0 ? "transparent" : `${COLORS.border}18` }}>
                <td style={{ padding: "10px 12px", fontWeight: 700, color: COLORS.text, fontFamily: "'Inter', -apple-system, sans-serif" }}>{c.flag} {c.name}</td>
                <td style={{ padding: "10px 12px", color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>{c.meOilDep}</td>
                <td style={{ padding: "10px 12px", color: COLORS.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>{c.hormuzDep}</td>
                <td style={{ padding: "10px 12px", color: COLORS.textMuted, fontFamily: "'Inter', -apple-system, sans-serif", fontSize: 13 }}>{c.reserves}</td>
                <td style={{ padding: "10px 12px" }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: c.color }} />
                    <span style={{ color: c.color, fontWeight: 700, fontSize: 13, fontFamily: "'Inter', -apple-system, sans-serif" }}>{c.riskLabel}</span>
                    <span style={{ width: 60, height: 3, borderRadius: 2, background: COLORS.border, display: "inline-block", position: "relative", overflow: "hidden" }}>
                      <span style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${c.risk * 20}%`, background: c.color, borderRadius: 2 }} />
                    </span>
                  </span>
                </td>
                <td style={{ padding: "10px 12px" }}>
                  {(() => {
                    const cbColor = c.cbPolicy === "HIKE" ? COLORS.accent : c.cbPolicy === "CUT" ? COLORS.emerald : COLORS.textDim;
                    return (
                      <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: 999, fontSize: 11, fontWeight: 700, letterSpacing: 0.5, textTransform: "uppercase", fontFamily: "'Inter', -apple-system, sans-serif", color: cbColor, background: `${cbColor}18`, border: `1px solid ${cbColor}33` }}>
                        {c.cbPolicy}
                      </span>
                    );
                  })()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <SourceTag sources={GROUNDED_DATA.countrySources} />
    </CardContainer>
  );
}

function ChinaDeepDivePanel() {
  const d = GROUNDED_DATA.chinaDeep;
  return (
    <CardContainer title="China Energy Exposure — Deep Dive" accent={COLORS.accent}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14, marginBottom: 14 }}>
        <Metric label="Iran Crude Imports" value={`${d.iranCrudeImportsMbd} mbd`} sub={`${d.iranShareOfImports} of total`} color={COLORS.accent} />
        <Metric label="ME Seaborne Share" value={d.meShareOfSeaborne.value} sub={`of China's seaborne crude (${d.meShareOfSeaborne.detail})`} color={COLORS.amber} />
        <Metric label="Strategic Reserves" value={`${d.sprBillionBbl}B bbl`} sub={`${d.sprCoverDays} days of imports`} color={COLORS.emerald} />
      </div>
      <div style={{ background: `${COLORS.blue}11`, borderRadius: 8, padding: 12, border: `1px solid ${COLORS.blue}22`, fontSize: 13, lineHeight: 1.6, color: COLORS.textMuted, marginBottom: 8, fontFamily: "'Inter', -apple-system, sans-serif" }}>
        <span style={{ fontWeight: 700, color: COLORS.blue, fontFamily: "'Inter', -apple-system, sans-serif" }}>Actions & Strategy: </span>
        {d.actions} {d.russianPivot}.
      </div>
      <SourceTag sources={d.sources} />
    </CardContainer>
  );
}

function MindmapPanel() {
  const [expandedLayer, setExpandedLayer] = useState(null);
  const layers = [
    { key: "military", label: "Military / Security", icon: "⚔️", color: COLORS.accent, nodes: GROUNDED_DATA.mindmapNodes.military },
    { key: "energy", label: "Energy / Commodity", icon: "🛢️", color: COLORS.amber, nodes: GROUNDED_DATA.mindmapNodes.energy },
    { key: "trade", label: "Trade / Maritime", icon: "🚢", color: COLORS.blue, nodes: GROUNDED_DATA.mindmapNodes.trade },
    { key: "diplomatic", label: "Diplomatic / Political", icon: "🏛️", color: COLORS.purple, nodes: GROUNDED_DATA.mindmapNodes.diplomatic },
    { key: "financial", label: "Financial / Macro", icon: "📉", color: COLORS.emerald, nodes: GROUNDED_DATA.mindmapNodes.financial },
  ];

  const confidenceColor = (c) => c >= 0.85 ? COLORS.accent : c >= 0.70 ? COLORS.amber : COLORS.blue;
  const horizonLabel = { immediate: "NOW", medium: "WEEKS", structural: "MONTHS" };
  const horizonColor = { immediate: COLORS.accent, medium: COLORS.amber, structural: COLORS.blue };

  return (
    <CardContainer title="Geopolitical Causal Mindmap" badge="Extended Reasoning" accent={COLORS.purple} style={{ gridColumn: "1 / -1" }}>
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <div style={{ display: "inline-block", background: `${COLORS.accent}22`, border: `2px solid ${COLORS.accent}`, borderRadius: 12, padding: "12px 28px" }}>
          <span style={{ fontSize: 16, fontWeight: 800, color: COLORS.accent, letterSpacing: 1, fontFamily: "'Inter', -apple-system, sans-serif" }}>🔴 IRAN ESCALATION</span>
          <div style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 2, fontFamily: "'JetBrains Mono', monospace" }}>Operation Epic Fury — Day 19 — Mar 18, 2026 08:04 UTC</div>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "center", gap: 4, marginBottom: 16 }}>
        {[...Array(5)].map((_, i) => <div key={i} style={{ width: 1, height: 24, background: `${COLORS.textDim}44` }} />)}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
        {layers.map((layer) => {
          const isExpanded = expandedLayer === layer.key;
          return (
            <div key={layer.key} onClick={() => setExpandedLayer(isExpanded ? null : layer.key)} style={{ cursor: "pointer", background: isExpanded ? `${layer.color}12` : COLORS.bg, border: `1px solid ${isExpanded ? layer.color : COLORS.border}`, borderRadius: 10, padding: 12, transition: "all 0.2s" }}>
              <div style={{ textAlign: "center", marginBottom: 10 }}>
                <span style={{ fontSize: 18 }}>{layer.icon}</span>
                <div style={{ fontSize: 12, fontWeight: 700, color: layer.color, marginTop: 4, textTransform: "uppercase", letterSpacing: 0.8, fontFamily: "'Inter', -apple-system, sans-serif" }}>{layer.label}</div>
              </div>
              {layer.nodes.map((node) => (
                <div key={node.id} style={{ background: isExpanded ? `${layer.color}0d` : COLORS.card, borderRadius: 6, padding: "8px 10px", marginBottom: 6, border: `1px solid ${isExpanded ? `${layer.color}33` : COLORS.border}`, transition: "all 0.2s" }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: COLORS.text, marginBottom: isExpanded ? 4 : 0, fontFamily: "'Inter', -apple-system, sans-serif" }}>{node.label}</div>
                  {isExpanded && (
                    <>
                      <div style={{ fontSize: 12, color: COLORS.textMuted, lineHeight: 1.4, marginBottom: 6, fontFamily: "'Inter', -apple-system, sans-serif" }}>{node.detail}</div>
                      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: confidenceColor(node.confidence), background: `${confidenceColor(node.confidence)}18`, borderRadius: 4, padding: "2px 6px", fontFamily: "'JetBrains Mono', monospace" }}>
                          {Math.round(node.confidence * 100)}% conf
                        </span>
                        <span style={{ fontSize: 11, fontWeight: 700, color: horizonColor[node.timeHorizon], background: `${horizonColor[node.timeHorizon]}18`, borderRadius: 4, padding: "2px 6px", fontFamily: "'Inter', -apple-system, sans-serif" }}>
                          {horizonLabel[node.timeHorizon]}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              ))}
              <div style={{ textAlign: "center", marginTop: 6 }}>
                <span style={{ fontSize: 11, color: COLORS.textDim, fontFamily: "'Inter', -apple-system, sans-serif" }}>{isExpanded ? "▲ collapse" : "▼ expand"}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: 16, padding: 12, background: `${COLORS.purple}11`, borderRadius: 8, border: `1px solid ${COLORS.purple}22`, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
        {[
          { label: "Confidence Key", items: [{ c: ">85%", col: COLORS.accent, t: "High confidence" }, { c: "70–84%", col: COLORS.amber, t: "Medium" }, { c: "<70%", col: COLORS.blue, t: "Speculative" }] },
          { label: "Time Horizon", items: [{ c: "NOW", col: COLORS.accent, t: "Hours–days" }, { c: "WEEKS", col: COLORS.amber, t: "Weeks–month" }, { c: "MONTHS", col: COLORS.blue, t: "Months–quarters" }] },
          { label: "Key Feedback Loops", items: [{ c: "NATO coalition friction", col: COLORS.accent, t: "→ Mine-hunting ships in Asia → months to clear → $150+" }, { c: "47% no cuts 2026 (FT)", col: COLORS.amber, t: "→ GS: 0.3% GDP hit → inflation +0.5–0.6% → stagflation" }, { c: "Lebanon ground ops", col: COLORS.accent, t: "→ War expanding → Houthi trigger → dual chokepoint → ~30% crude" }] },
        ].map((section) => (
          <div key={section.label}>
            <div style={{ fontSize: 11, fontWeight: 700, color: COLORS.purple, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6, fontFamily: "'Inter', -apple-system, sans-serif" }}>{section.label}</div>
            {section.items.map((item) => (
              <div key={item.c} style={{ fontSize: 11, color: COLORS.textMuted, marginBottom: 2, fontFamily: "'Inter', -apple-system, sans-serif" }}>
                <span style={{ color: item.col, fontWeight: 700 }}>{item.c}</span> {item.t}
              </div>
            ))}
          </div>
        ))}
      </div>
    </CardContainer>
  );
}

function TimelinePanel() {
  const events = GROUNDED_DATA.timeline;
  return (
    <CardContainer title="Conflict Timeline & Oil Impact" accent={COLORS.blue} style={{ gridColumn: "1 / -1" }}>
      <div style={{ position: "relative", paddingLeft: 20 }}>
        <div style={{ position: "absolute", left: 6, top: 0, bottom: 0, width: 2, background: COLORS.border }} />
        {events.map((ev, i) => {
          const isRecent = i >= events.length - 4;
          const isCurrent = i === events.length - 1;
          return (
            <div key={i} style={{ position: "relative", paddingLeft: 24, paddingBottom: 14, marginBottom: 2 }}>
              <div style={{ position: "absolute", left: -2, top: 4, width: 10, height: 10, borderRadius: "50%", background: isCurrent ? COLORS.accent : isRecent ? COLORS.amber : COLORS.blue, border: `2px solid ${COLORS.bg}`, boxShadow: isCurrent ? `0 0 10px ${COLORS.accent}` : isRecent ? `0 0 6px ${COLORS.amber}66` : "none" }} />
              <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                <span style={{ fontSize: 13, fontWeight: 800, color: isCurrent ? COLORS.accent : isRecent ? COLORS.amber : COLORS.blue, minWidth: 100, fontFamily: "'JetBrains Mono', monospace" }}>{ev.date}</span>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: COLORS.text, fontFamily: "'Inter', -apple-system, sans-serif" }}>{ev.event}</div>
                  <div style={{ fontSize: 13, color: COLORS.textMuted, marginTop: 2, fontFamily: "'Inter', -apple-system, sans-serif" }}>Oil: {ev.oilImpact}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <SourceTag sources={GROUNDED_DATA.timelineSources} />
    </CardContainer>
  );
}

function GoldmanAnalysisPanel() {
  const g = GROUNDED_DATA.goldmanAnalysis;
  return (
    <CardContainer title="Analyst Scenario Analysis" accent={COLORS.emerald}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 12 }}>
        <Metric label="GS Risk Premium" value={g.riskPremium} sub={g.riskPremiumPct} color={COLORS.amber} />
        <Metric label="GS Base Case" value={g.q2Forecast.value} sub={g.q2Forecast.detail} color={COLORS.blue} />
        <Metric label="GS Upside" value={g.upside.value} sub={g.upside.detail} color={COLORS.accent} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {[
          { scenario: "GS Base (21-Day Disruption)", brent: g.q4Forecast.value, detail: g.q4Forecast.detail + ". GS: recession probability 25%. Pentagon 4–6 weeks extends beyond GS base case.", color: COLORS.blue },
          { scenario: "GS 30-Day / Adverse", brent: "$130 peak", detail: "GS 'adverse' scenario: oil peaks $130. ING: 'market highs still ahead of us.' Capital Economics: 3-month conflict → $150. GS: eurozone inflation could peak 4.4%. RBC: conflict could last 'well into the spring' (FT). Barclays: 'Trump put' may erode if oil sustainably >$100.", color: COLORS.amber },
          { scenario: "GS 60-Day / Very Adverse", brent: "$150 peak", detail: "Goldman: 60-day = Q4 at $93/$89, oil peaks $150. GS: $145+ if flows at current levels. Iran FM: $200/bbl. Rystad worst-case: ME crude could fall to 6M bpd (70% drop). Morgan Stanley: oil doubling could shave 1.5% off US GDP. CRS: reopening could take months.", color: COLORS.accent },
        ].map((s) => (
          <div key={s.scenario} style={{ background: `${s.color}0d`, borderRadius: 8, padding: 12, border: `1px solid ${s.color}22` }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: s.color, fontFamily: "'Inter', -apple-system, sans-serif" }}>{s.scenario}</span>
              <span style={{ fontSize: 14, fontWeight: 800, color: COLORS.text, fontFamily: "'JetBrains Mono', monospace" }}>{s.brent}</span>
            </div>
            <div style={{ fontSize: 13, color: COLORS.textMuted, lineHeight: 1.4, fontFamily: "'Inter', -apple-system, sans-serif" }}>{s.detail}</div>
          </div>
        ))}
      </div>
      <SourceTag sources={g.sources} />
    </CardContainer>
  );
}

// ── Main Dashboard ──

export default function IranGeopolDashboard() {
  const [activeTab, setActiveTab] = useState("overview");

  const tabs = [
    { key: "overview", label: "Situation Overview" },
    { key: "countries", label: "Country Exposure" },
    { key: "mindmap", label: "Geopolitical Mindmap" },
    { key: "timeline", label: "Conflict Timeline" },
  ];

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text, fontFamily: "'Inter', -apple-system, sans-serif" }}>
      {/* Frozen cookbook snapshot — not a live feed; see MCP_Dashboard_Demo/README.md */}
      <div
        role="status"
        style={{
          padding: "10px 28px",
          fontSize: 12,
          lineHeight: 1.45,
          color: COLORS.text,
          background: `linear-gradient(90deg, ${COLORS.amberGlow} 0%, ${COLORS.blueGlow} 100%)`,
          borderBottom: `1px solid ${COLORS.border}`,
          fontFamily: "'Inter', -apple-system, sans-serif",
        }}
      >
        <strong style={{ color: COLORS.amber }}>Illustration snapshot</strong>
        {" — "}
        Data is baked into <code style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>GROUNDED_DATA</code> (frozen <strong>2026-03-18</strong>). This UI does not call Bigdata.com in the browser. Production refresh & hosting run elsewhere; see repo README.
      </div>
      {/* Header */}
      <div style={{ borderBottom: `1px solid ${COLORS.border}`, padding: "16px 28px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ width: 10, height: 10, borderRadius: "50%", background: COLORS.accent, boxShadow: `0 0 12px ${COLORS.accent}88`, animation: "pulse 2s infinite" }} />
          <div>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, letterSpacing: -0.3, fontFamily: "'Inter', -apple-system, sans-serif" }}>Iran Geopolitical Risk Intelligence Dashboard</h1>
            <span style={{ fontSize: 12, color: COLORS.textDim, fontFamily: "'Inter', -apple-system, sans-serif" }}>Country-Level Exposure & Causal Analysis — Grounded with <a href="https://bigdata.com" style={{ color: COLORS.blue, textDecoration: "none" }}>Bigdata.com</a></span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <Badge color={COLORS.accent} glow={COLORS.accentGlow}>Brent $102.10</Badge>
          <span style={{ fontSize: 13, color: COLORS.textDim, fontFamily: "'JetBrains Mono', monospace" }}>Mar 18, 2026 — 08:04 UTC</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: `1px solid ${COLORS.border}`, padding: "0 28px" }}>
        {tabs.map((tab) => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key)} style={{
            padding: "12px 20px", fontSize: 13, fontWeight: 700, cursor: "pointer", border: "none",
            background: "transparent", color: activeTab === tab.key ? COLORS.text : COLORS.textDim,
            borderBottom: activeTab === tab.key ? `2px solid ${COLORS.blue}` : "2px solid transparent",
            letterSpacing: 0.5, transition: "all 0.15s", textTransform: "uppercase", fontFamily: "'Inter', -apple-system, sans-serif",
          }}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "20px 28px", maxWidth: 1300, margin: "0 auto" }}>

        {activeTab === "overview" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <EnergyMarketsPanel />
            <HormuzStatusPanel />
            <ChinaDeepDivePanel />
            <GoldmanAnalysisPanel />
          </div>
        )}

        {activeTab === "countries" && (
          <div style={{ display: "grid", gap: 16 }}>
            <CountryExposurePanel />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <ChinaDeepDivePanel />
              <GoldmanAnalysisPanel />
            </div>
          </div>
        )}

        {activeTab === "mindmap" && (
          <div style={{ display: "grid", gap: 16 }}>
            <MindmapPanel />
          </div>
        )}

        {activeTab === "timeline" && (
          <div style={{ display: "grid", gap: 16 }}>
            <TimelinePanel />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <EnergyMarketsPanel />
              <HormuzStatusPanel />
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ borderTop: `1px solid ${COLORS.border}`, padding: "14px 28px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 12, color: COLORS.textDim, fontFamily: "'Inter', -apple-system, sans-serif" }}>All data grounded via <a href="https://bigdata.com" style={{ color: COLORS.blue, textDecoration: "none" }}>Bigdata.com</a> MCP — see individual panel sources for attribution. Last cycle: Mar 18, 2026 08:04 UTC.</span>
        <span style={{ fontSize: 12, color: COLORS.textDim, fontFamily: "'Inter', -apple-system, sans-serif" }}>Powered by <a href="https://bigdata.com" style={{ color: COLORS.blue, textDecoration: "none" }}>Bigdata.com</a></span>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        * { box-sizing: border-box; }
        button { font-family: inherit; }
        a:hover { opacity: 0.8; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0a0e1a; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
      `}</style>
    </div>
  );
}
