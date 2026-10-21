// Number formatting that matches Python's format(x, ".Nf"), so the page shows the same
// digits as the dashboard. Python rounds an exact tie (0.625 to 2 places) to even, "0.62";
// toFixed rounds it up, "0.63". Doubles at this scale are ~1e-16 apart, so 20 extra digits
// reveal whether x sits exactly on a tie. tests/test_site_format.py checks this vs Python.

export function fixed(x, places) {
  if (Object.is(x, -0)) return "-" + (0).toFixed(places); // Python keeps the sign of -0.0
  const wide = Math.abs(x).toFixed(places + 20);
  const [whole, frac = ""] = wide.split(".");
  const tail = frac.slice(places);
  const out = x.toFixed(places);
  if (tail !== "5" + "0".repeat(19)) return out;
  const lastDigit = Number((places ? frac[places - 1] : whole.at(-1)) ?? 0);
  if (lastDigit % 2 === 1) return out; // odd: rounding up lands on even, as toFixed did
  const kept = places ? `${whole}.${frac.slice(0, places)}` : whole;
  return (x < 0 ? "-" : "") + kept;
}

// Python's "+.Nf": a sign on every value, including "+0.00" and "-0.00".
export function signed(x, places) {
  const s = fixed(x, places);
  return s.startsWith("-") ? s : "+" + s;
}

export function integer(n) {
  return Math.round(n).toLocaleString("en-US");
}
