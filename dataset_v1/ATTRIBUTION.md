# Dataset attribution

This directory contains transformations of two Kaggle datasets distributed by
their publishers under the Creative Commons Attribution 4.0 International
license. Both were accessed on 2026-08-11.

1. Ariba Shafaqat, **Plants Growth and Care Recommendations**,
   [Kaggle dataset page](https://www.kaggle.com/datasets/aribashafaqat/plants-growth-and-care-recommendations/versions/1),
   CC BY 4.0. Modifications: CP1252 decoding, whitespace normalization, removal
   of four exact duplicate rows, whole-entity exclusion when duplicate names
   contain conflicting fields, atomic-entry transformation, stable IDs, and
   template contract queries.
2. Souvik Rana, **Indoor Plant Health & Growth Dataset**,
   [Kaggle dataset page](https://www.kaggle.com/datasets/souvikrana17/indoor-plant-health-and-growth-dataset),
   CC BY 4.0. Modifications: removal of rows with contradictory pest presence
   and severity, type normalization, stable IDs, and explicit use restrictions.

License text and terms: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The rejected `prakash27x` dataset is documented for audit reproducibility only.
Its contents are not redistributed or used in processed outputs because its
Kaggle license is Unknown.
