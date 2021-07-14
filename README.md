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


## Example


Notebook 1 ) 
Notebook 2) 
## Docker



## Requirements

- Python 3.7 and above.


## License

Under MIT license. See [LICENSE](LICENSE).

## Authors

- Oozge Oozguc
- Varun Kapoor <randomaccessiblekapoor@gmail.com>

