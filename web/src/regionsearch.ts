import type { Region } from "./irh";

/** For matching: no case, no accents, single spaces. */
export const normalise = (name: string) =>
  name.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/\s+/g, " ").trim();

/** Without spaces, hyphens or other signs: "J-Jpp" and "jjpp" are the same. */
const compact = (name: string) => normalise(name).replace(/[^a-z0-9]/g, "");

const KIND_ORDER = { "ice sheet": 0, basin: 1, place: 2 };

export interface Suggestion {
  region: Region;
  matched: string | null; // the name that matched, when not in the label shown (an alias, an IMBIE code)
}

/**
 * The search of the region filter: every name of a region (its label, short name, name and
 * aliases, e.g. the IMBIE code or the names of a place in other languages), matched on any part,
 * not only the start as a <datalist> does in Firefox. Best first: the whole name, its start, the
 * start of a word, any part; then ice sheets, basins, places; then the regions with more lines.
 */
export function regionSearch(regions: Region[], label: (r: Region) => string): (text: string, limit?: number) => Suggestion[] {
  const searchable = regions.map((region) => {
    const shown = label(region);
    const names = [shown, ...[region.key, region.name, ...region.aliases].filter((n): n is string => !!n)];
    return {
      region,
      shown: normalise(shown),
      names: names.map((n) => ({ text: n, flat: compact(n), words: normalise(n).split(/[^a-z0-9]+/).filter(Boolean) })),
    };
  });
  return (text, limit = 8) => {
    const q = compact(text);
    if (!q) return [];
    const found: (Suggestion & { rank: number })[] = [];
    for (const { region, shown, names } of searchable) {
      let rank = Infinity;
      let matched = "";
      for (const n of names) {
        const r = n.flat === q ? 0 : n.flat.startsWith(q) ? 1 : n.words.some((w) => w.startsWith(q)) ? 2 : n.flat.includes(q) ? 3 : Infinity;
        if (r < rank) [rank, matched] = [r, n.text];
      }
      if (rank === Infinity) continue;
      found.push({ region, matched: shown.includes(normalise(matched)) ? null : matched, rank });
    }
    found.sort(
      (a, b) =>
        a.rank - b.rank ||
        KIND_ORDER[a.region.kind] - KIND_ORDER[b.region.kind] ||
        b.region.tracks.length - a.region.tracks.length ||
        label(a.region).localeCompare(label(b.region)),
    );
    return found.slice(0, limit).map(({ region, matched }) => ({ region, matched }));
  };
}
