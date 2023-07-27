# eROSITA Lockman Hole source classification

This repository accompanies the paper "SRG/eROSITA Survey in the Lockman Hole: Classification of X-ray Sources" (Belvedersky, Bykov, Gilfanov; 2022) published in Astronomy Letters.

Binary classification (extragalactic or not) of the eROSITA LH sources; cross-match with spectral catalogs and databases; creation procedure of the eROSITA Lockman Hole catalog.

### Requirements

Python 3.9 or higher

### Installation

```
pip3 install git+https://github.com/mbelveder/ero-lh-class.git
```

### Input data

The input data will be available after the Lockman Hole catalog is published.

### Usage:

```
lh-class-pipeline -input_path /path/to/downloaded/data/ -output_path /path/to/save/result/data/
```
or just

```
lh-class-pipeline -i /path/to/downloaded/data/ -o /path/to/save/result/data/
```

Note: the output directory will be created (with parents) if not exist already.

### Building blocks

- **lh-class**: cross-match with the spectral catalogs and databases for the nnmag catalog (Bykov, Belvedersky, Gilfanov; 2022).

- **lh-srgz-perp**: SRGz preprocessing. As a result, evry nnmag counterpart has SRGz features AND every X-ray source has two version of optical counterparts (nnmag and SRGz). Mostly these two counterparts is the same optical source, but not all of them.

- **lh-srgz-spec**: cross-match with spectral catalogs and databases for the SRGz catalogs (Mescheryakov et al, in preparation), similar to the `lh-class`.

- **lh-postprocess**: join all the auxiliary tables into sinle catalog, columns renaming and reduction.

---

- **lh-class-pipeline**: a procedure that combine all the above scripts to run the whole pipeline with one command.
