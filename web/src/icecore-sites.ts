/**
 * The ice-core sites shown in the 3D viewer (icecores.ts), without three.js: also the cores of the
 * submission form.
 */
export interface IceCore {
  name: string; // label of the flag
  site: string;
  lat: number;
  lon: number;
  elevation: number; // m above sea level, of the surface at the drill site
  depth: number; // m, drilled depth (to the bed for most)
  bottomAge: string | null; // oldest dated ice, as published
  drilled: string; // field seasons
  reference: string;
  poleScale?: number; // taller pole, so the flag clears a neighbour's
}

export const ICE_CORES: IceCore[] = [
  { name: "EDC", site: "EPICA Dome C", lat: -75.1017, lon: 123.3482, elevation: 3233, depth: 3270, bottomAge: "≈ 800 ka", drilled: "1996–2004", reference: "EPICA community members (2004), Nature" },
  { name: "BELDC", site: "Beyond EPICA, Little Dome C", lat: -75.2992, lon: 122.4451, elevation: 3230, depth: 2800, bottomAge: "> 1.2 Ma", drilled: "2021–2025", reference: "Beyond EPICA – Oldest Ice (2025); Lilien et al. (2021), The Cryosphere", poleScale: 2.2 },
  { name: "Dome Fuji", site: "Dome Fuji (DF2)", lat: -77.3167, lon: 39.7, elevation: 3810, depth: 3035, bottomAge: "≈ 720 ka", drilled: "2003–2007", reference: "Dome Fuji Ice Core Project Members (2017), Science Advances" },
  { name: "Vostok", site: "Vostok (5G)", lat: -78.4644, lon: 106.8372, elevation: 3488, depth: 3769, bottomAge: "≈ 420 ka (meteoric ice)", drilled: "1990–2012", reference: "Petit et al. (1999), Nature; Vasiliev et al. (2012)" },
  { name: "EDML", site: "EPICA Dronning Maud Land (Kohnen)", lat: -75.0025, lon: 0.0684, elevation: 2892, depth: 2774, bottomAge: "≈ 150 ka (undisturbed record)", drilled: "2001–2006", reference: "EPICA community members (2006), Nature" },
  { name: "TALDICE", site: "Talos Dome", lat: -72.8167, lon: 159.1833, elevation: 2315, depth: 1620, bottomAge: "≈ 250 ka", drilled: "2004–2008", reference: "Stenni et al. (2011), Nature Geoscience; Buiron et al. (2011), Climate of the Past" },
  { name: "Taylor Dome", site: "Taylor Dome", lat: -77.7964, lon: 158.7239, elevation: 2365, depth: 554, bottomAge: "≈ 130 ka", drilled: "1993–1994", reference: "Steig et al. (2000), Geografiska Annaler" },
  { name: "Law Dome", site: "Law Dome (DSS)", lat: -66.7725, lon: 112.8113, elevation: 1370, depth: 1196, bottomAge: null, drilled: "1988–1993", reference: "Morgan et al. (1997), Journal of Glaciology" },
  { name: "SPICEcore", site: "South Pole", lat: -89.99, lon: -98.16, elevation: 2835, depth: 1751, bottomAge: "≈ 54 ka", drilled: "2014–2016", reference: "Winski et al. (2019), Climate of the Past" },
  { name: "WDC", site: "WAIS Divide (WDC06A)", lat: -79.467, lon: -112.085, elevation: 1766, depth: 3405, bottomAge: "≈ 68 ka", drilled: "2006–2011", reference: "WAIS Divide Project Members (2013), Nature" },
  { name: "Byrd", site: "Byrd Station", lat: -80.0167, lon: -119.5167, elevation: 1530, depth: 2164, bottomAge: null, drilled: "1966–1968", reference: "Ueda and Garfield (1969), CRREL" },
  { name: "Siple Dome", site: "Siple Dome A", lat: -81.65, lon: -148.81, elevation: 621, depth: 1004, bottomAge: "≈ 100 ka", drilled: "1996–1999", reference: "Taylor et al. (2004), Journal of Glaciology" },
  { name: "Skytrain", site: "Skytrain Ice Rise", lat: -79.741, lon: -78.5448, elevation: 784, depth: 651, bottomAge: "≈ 130 ka", drilled: "2018–2019", reference: "Mulvaney et al. (2023), Climate of the Past" },
];

// Other sites, not shown: add them back to ICE_CORES to show them
export const OTHER_ICE_CORES: IceCore[] = [
  { name: "ABN", site: "Aurora Basin North", lat: -71.17, lon: 111.37, elevation: 2690, depth: 303, bottomAge: null, drilled: "2013–2014", reference: "Servettaz et al. (2023), Climate of the Past" },
  { name: "RICE", site: "Roosevelt Island (RICE)", lat: -79.36, lon: -161.71, elevation: 550, depth: 763, bottomAge: "≈ 83 ka", drilled: "2011–2013", reference: "Lee et al. (2020), Climate of the Past" },
  { name: "Fletcher", site: "Fletcher Promontory", lat: -77.9, lon: -82.61, elevation: 873, depth: 654, bottomAge: null, drilled: "2011–2012", reference: "Mulvaney et al. (2014), Annals of Glaciology" },
  { name: "Berkner", site: "Berkner Island (Thyssenhöhe)", lat: -79.55, lon: -45.68, elevation: 890, depth: 948, bottomAge: null, drilled: "2002–2005", reference: "Mulvaney et al. (2007), Annals of Glaciology" },
  { name: "JRI", site: "James Ross Island (Mount Haddington)", lat: -64.2017, lon: -57.685, elevation: 1542, depth: 363, bottomAge: null, drilled: "2008", reference: "Mulvaney et al. (2012), Nature" },
];
