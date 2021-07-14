# PeCoWaCo


This python package is to be used in conjugation with the Fiji plugin WizardofOz to analyze periodic contractions of cortical waves, which travel around blastomeres in an oscillatory fashion. 

Download Fiji from [Fiji](https://imagej.github.io/), After download open Fiji and to install the Fiji plugin WizardofOz check the update cite MTrack. 

Click Help ▶ Update....

Click the Manage update sites button.

Select the MTrack update site in the list.

Click Close and then click Apply changes.

Restart Fiji.

Launch the plugin with Plugins ▶ WizardofOz ▶ Local Deformation & Intensity Tracking.

In the plugin the Raw image (shown below) and its segmentation image are the inputs. In the plugin we fit circles for computing the local curvature
information using the start, center and end
point of a 10 μm strip on the cell surface to fit
a circle. The strip is then moved by 1 pixel
along the segmented cell and a new circle is
fitted. This process is repeated till all the
points of the cell are covered. The radius of
curvature of the 10 μm strip boundaries are
averaged. Kymograph of local curvature
values around the perimeter over time is
produced by plotting the perimeter of the strip
over time.
Curvature kymographs obtained from local
curvature tracking are then exported into this python package (pecowaco) for amplitude, frequency and velocity of wave analysis. 

![Notebook Description](https://github.com/kapoorlab/PeCoWaCo/blob/main/Images/PastedGraphic.png)

## Installation
This package can be installed with:

`pip install --user PeCoWaCo`

If you are building this from the source, clone the repository and install via

```bash
git clone https://github.com/MechaBlasto/PeCoWaCo/

cd PeCoWaCo

pip install --user -e .

# or, to install in editable mode AND grab all of the developer tools
# (this is required if you want to contribute code back to NapaTrackMater)
pip install --user -r requirements.txt
```

### Pipenv install

Pipenv allows you to install dependencies in a virtual environment.

```bash
# install pipenv if you don't already have it installed
pip install --user pipenv

# clone the repository and sync the dependencies
git clone https://github.com/MechaBlasto/PeCoWaCo/
cd PeCoWaCo
pipenv sync

# make the current package available
pipenv run python setup.py develop

# you can run the example notebooks by starting the jupyter notebook inside the virtual env
pipenv run jupyter notebook
```

Access the `example` folder and run the cells.


## Usage
The WizardofOz plugin can be run using a single channel image or a dual channel image. The segmentation image for both is the same. In the single channel mode curvature, distance and intensity of the membrane is computed, in dual channel mode the intensity along the membrane of the second channel is computed as well. We provide two notebooks for analysis of these two modes.

## Example


[Notebook](https://github.com/kapoorlab/PeCoWaCo/blob/main/examples/OscillationQuantifier_3kymo.ipynb) 1)  In this notebook, the curvature, distance and intensity kymograph output coming from a single channel of WoZ plugin serves as an input. Users can choose the start and the end points along the x and y axis of kymograph to exclude certain regions from further computation. We calculate the root mean square, space resolved FFT along the time axis and a space averaged FFT plot with interactive bokeh plots. All the information is saved as csv files.

[Notebook](https://github.com/kapoorlab/PeCoWaCo/blob/main/examples/OscillationQuantifier_4kymo.ipynb) 2) In this notebook, the curvature, distance and intensity kymograph output coming from dual channel mode of WoZ plugin serves as an input. We compute the same quantities as in notebook 1 but in addition now with the intnesity kymograph analysis of the second channel included as well.



## Requirements

- Python 3.7 and above.


## License

Under MIT license. See [LICENSE](LICENSE).

## Authors

- Oozge Oozguc
- Varun Kapoor <randomaccessiblekapoor@gmail.com>

