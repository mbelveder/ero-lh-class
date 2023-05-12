# eROSITA Lockman Hole source classification

Binary classification (extragalactic or not) of the eROSITA LH sources; cross-match with spectral catalogs and databases; creation procedure of the eROSITA Lockman Hole catalog.

### Requirements

Python 3.9 or higher

### Installation

`pip3 install git+https://github.com/mbelveder/lh.git`

### Input data

[Download](https://disk.yandex.ru/d/F_Q55KtS36gV8A)

TODO: describe the content

### Usage:

```
lh-class-pipeline -input_path /path/to/the/downloaded/dir -output_path /path/to/save/auxiliary/and/result/data
```
### Building blocks

- **lh-class**: cross-match with spectral catalogs and databases for the nnmag catalog (Bykov, Belvedersky, Gilfanov; 2022).

- **lh-srgz-perp**: SRGz preprocessing. As a esult evry nnmag counterpart has SRGz features AND every X-ray source has two version of optical counterparts (nnmag and SRGz). Mostly these two counterparts is the same optical source, but not all of them.

- **lh-srgz-spec** cross-match with spectral catalogs and databases for the SRGz catalogs (Mescheryakov et al, in preparation), similar to the lh-class.

- **lh-postprocess** join all the auxiliary tables into sinle catalog, columns renaming and reduction.

---

- **lh-class-pipeline** a procedure that combine all the above scripts to run the whole pipeline with one command.
