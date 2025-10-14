the working script is `create-2d-doc-dataset.py`
you need to get a JSESSIONID from dossier facile back office, and export it as env `DOSSIER_FACILE_JSESSIONID` to run the script.
the script assumes you have a `../../datasets/2d-doc/tax-notices.csv` file which corresponds to a list of dossier facile's file_id (avis d'impositions) from metabase.
Refer to the notebook for more details.

This script will :
1) extract all file_id from the tax-notices.csv.
2) will batch process those file_ids and do the next things:
    a) download the real file using DOSSIER_FACILE_JSESSIONID
    b) try to find a 2dDoc and extract it's raw data using pylibdmtx
    c) if a 2dDoc is found, parse it's content using https://github.com/nipo/tdd (clone with submodules) and put it into a datastructure (TaxNoticeData)
3) gather all the valid TaxNoticeData as a list, and save them to `datasets/2d-doc/tax-notices-extracted-2d-doc.csv`