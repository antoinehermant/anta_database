/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Base URL of the exported data, e.g. the bucket. Defaults to <base>/data
  readonly VITE_DATA_URL?: string;
  // Base URL of the isochrones only (<url>/irh), for the development viewer of a version of the
  // data not published yet. Defaults to VITE_DATA_URL
  readonly VITE_IRH_DATA_URL?: string;
  // Set: the development viewer, with a banner, and not indexed by search engines
  readonly VITE_DEV_VIEWER?: string;
  // Published database (prefix of manifest.json), for the download page. Defaults to the bucket
  readonly VITE_DB_URL?: string;
  // Submission portal (portal/app.py), for the submit page. Unset: same origin in development,
  // and submissions closed in a build
  readonly VITE_PORTAL_URL?: string;
  // "true" once the submission portal is public; before, only /submit/?dev uses it
  readonly VITE_PORTAL_PUBLIC?: string;
}
