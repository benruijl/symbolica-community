# Tutorial data

`scalars_2p_3p.json` is the normalized scalar-model fixture from GammaLoop's
FeynKit branch at revision `1c61bb34811090bd1d8ed9cda81a5d85d477e527`. It is
bundled here so the core tutorials are deterministic and do not require an
external UFO model or Python UFO loader.

`sm.json` is a byte-for-byte copy of
`assets/models/json/sm/sm.json` from GammaLoop's FeynKit branch at revision
`e5b75ffb06844c3228d607032423bc3ffaf71573`. Its SHA-256 digest is
`8ba8e7f6f6271e47ee9ba8a41b2a090b6183924af03226a21d96c0846815d3b1`.
The one-loop QCD numerator tutorial uses this authoritative normalized Standard
Model fixture without requiring the optional UFO loader.

`ufo_scalars/` is the corresponding minimal raw UFO source: the package modules
needed by `ufo-model-loader` plus its default restriction card. It omits unused
Standard-Model compatibility parameters from the larger upstream fixture while
preserving the scalar particles and interactions used throughout the tutorials.
The dedicated
`scripts/check_feynkit_ufo.py` smoke check sets the model's supported environment
switch to generate only two- and three-point vertices. Raw import is optional;
it requires Python 3.11 or newer and `ufo-model-loader>=0.1.6`, installed with
`pip install 'symbolica[feynkit-ufo]'`.
