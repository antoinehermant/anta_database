/**
 * What the submission form knows of radars and ice cores, from the data of the package
 * (anta_database/data), also read by the checks of the portal: the platform of each radar, and
 * the chronologies with the cores they date. The cores are those of the 3D viewer.
 */
import platforms from "../../../anta_database/data/radar_platforms.json";
import chronologies from "../../../anta_database/data/chronologies.json";
import { ICE_CORES, OTHER_ICE_CORES } from "../icecore-sites";

export type Platform = "airborne" | "ground-based";
export const PLATFORMS: [Platform, string][] = [
  ["airborne", "Airborne"],
  ["ground-based", "Ground-based"],
];

const byRadar = new Map(Object.entries(platforms.platforms).map(([radar, p]) => [radar.toLowerCase(), p as Platform]));

/** The platform of a radar the database knows, compared without case. */
export function platformOf(radar: string): Platform | undefined {
  return byRadar.get(radar.trim().toLowerCase());
}

/** Ice cores, by the name of the viewer, with their site: "EDC" (EPICA Dome C). */
export const CORES: [name: string, site: string][] = [...ICE_CORES, ...OTHER_ICE_CORES].map((c) => [c.name, c.site]);

const CHRONOLOGIES = chronologies.chronologies as Record<string, { cores: string[]; reference: string }>;

/** The chronologies dating a core (all of them for a core not known), with their reference. */
export function chronologiesOf(core: string): [name: string, reference: string][] {
  const known = Object.entries(CHRONOLOGIES);
  const name = core.trim().toLowerCase();
  const dating = known.filter(([, c]) => c.cores.some((k) => k.toLowerCase() === name));
  return (dating.length ? dating : known).map(([n, c]) => [n, c.reference]);
}

/** The cores a chronology dates (all the cores for a chronology not known), with their site. */
export function coresOf(chronology: string): [name: string, site: string][] {
  const entry = Object.entries(CHRONOLOGIES).find(([n]) => n.toLowerCase() === chronology.trim().toLowerCase());
  return entry ? CORES.filter(([name]) => entry[1].cores.includes(name)) : CORES;
}

/**
 * Why a chronology and a core are not a pair we know, when at least one of them is known (a known
 * chronology with a core it does not date, or a known core with a chronology we do not have for
 * it); null when they are a known pair, or when neither is known.
 */
export function chronologyMismatch(core: string, chronology: string): string | null {
  const entry = Object.entries(CHRONOLOGIES).find(([n]) => n.toLowerCase() === chronology.trim().toLowerCase());
  // The cores known to the checks of the portal too (the same list)
  const coreName = chronologies.cores.find((n) => n.toLowerCase() === core.trim().toLowerCase());
  if (entry) return entry[1].cores.includes(coreName ?? "") ? null : `${entry[0]} dates ${entry[1].cores.join(", ")}, not ${core.trim()}`;
  if (!coreName) return null;
  const dating = Object.entries(CHRONOLOGIES)
    .filter(([, c]) => c.cores.includes(coreName))
    .map(([n]) => n);
  return dating.length
    ? `we know ${coreName} dated by ${dating.join(", ")}, not by ${chronology.trim()}`
    : `we know no chronology of ${coreName}, so not ${chronology.trim()}`;
}
