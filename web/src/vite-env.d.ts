/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Base URL of the exported data, e.g. the bucket. Defaults to <base>/data
  readonly VITE_DATA_URL?: string;
}
