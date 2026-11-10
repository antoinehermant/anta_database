# Roadmap

Here is a todo list of features that we want to include in the AntADatabase.

## Important
- [x] Implement HDF5 format (In process, for next release)
- [x] Filter by radar system
- [x] Create shapefiles, e.g. for QAntarctica
- [ ] More familiar basin/region naming (e.g. Wilkes, Thwaites etc)
- [ ] Cloud host (some testing with cloud_database branch)
- [ ] Improved age assignment (currently hardcoded during raw data processing, but could be done in the Indexing so ages are updated simply by reindexing)
- [x] Sort airborne vs ground-based data: `platform` per line, `db.query(platform=...)` (v2)
- [x] How each layer was dated: `dating_method`, `ice_core`, `chronology` per layer, queryable (v2; values to fill in anta_json)
- [ ] Firn correction of each dataset in anta_json (`firn correction applied`, `firn correction`), from the articles;
      the submit portal asks it already

## To discuss
- [ ] A netCDF version 
- [ ] Implement method to date layers from closest/latest chronology
- [ ] Refine isochrone ages for consistent layers across the database
- [ ] On demand, ice thickness, bed and surface from the gridded Bedmap3 where neither the dataset nor
      the Bedmap survey lines have them, in their own variables marked as gridded (v2 keeps NaN there)
- [ ] One consistent firn correction for the whole harmonised database, once the firn correction of every
      dataset is known
