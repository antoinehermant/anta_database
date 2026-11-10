# Roadmap

Here is a todo list of features that we want to include in the AntADatabase.

## Important
- [x] Implement HDF5 format (In process, for next release)
- [x] Filter by radar system
- [x] Create shapefiles, e.g. for QAntarctica
- [ ] More familiar basin/region naming (e.g. Wilkes, Thwaites etc)
- [ ] Cloud host (some testing with cloud_database branch)
- [ ] Improved age assignment (currently hardcoded during raw data processing, but could be done in the Indexing so ages are updated simply by reindexing)
- [ ] Sort airborne vs ground-based data: a `platform` field per raw file, in progress for v2

## To discuss
- [ ] A netCDF version 
- [ ] Implement method to date layers from closest/latest chronology
- [ ] Refine isochrone ages for consistent layers across the database
- [ ] On demand, ice thickness, bed and surface from the gridded Bedmap3 where neither the dataset nor
      the Bedmap survey lines have them, in their own variables marked as gridded (v2 keeps NaN there)

