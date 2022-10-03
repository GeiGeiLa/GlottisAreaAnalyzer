.. _software:

Software setup
==============

.. role:: python(code)
    :language: python

We assume that Windows 10 is running on your system, and you have basic knowledge in Python.

Requirements
------------

Ensure that the following software is installed on your computer:

- Python 3.x, tested with Python 3.6 and Anaconda package
- IDT SDK (to access drivers, get from distributor)

Supported cameras
-----------------

OpenHSV currently provides only support for the CCM series of `IDT high-speed cameras`_, but can be extended to your individual case. 

Install and Operate OpenHSV
---------------------------

Clone our `Github repository`_ and install the :python:`openhsv` package with :python:`pip`:

.. code-block:: bash

    pip install setup.py

Run OpenHSV by executing the :python:`main.py` script.

.. code-block:: bash

    python main.py 

An easy way to create a shortcut to the OpenHSV software is to create a :python:`bat` file with the following content:

.. code-block:: bash

    # Activate anaconda3 environment
    call "path/to/activate.bat" "path/to/anaconda3"
    # Go to openhsv directory 
    pushd "path/to/openhsv" 
    # Execute main.py without showing console
    python.exe -u "main.py" 

You may then place a shortcut to that file on the Desktop to allow easy access for examiners.

A common problem is that the camera drivers are not found. Ensure that the driver files are either available in the system PATH or 
directly in the OpenHSV directory. You may need all DLLs from the IDT SDK. 

Testing environment
-------------------

If no camera is available or other parts of OpenHSV should be tested, we supply a dummy camera that loops through the example video shipped with OpenHSV.
You only need to change in the :python:`__init__.py` file the following line:

.. code-block:: python

    from openhsv.hardware.camera import IdtCamera as Camera

to

.. code-block:: python

    from openhsv.hardware.camera import DummyCamera as Camera

.. _`Github repository`: https://github.com/anki-xyz/openhsv
.. _`IDT high-speed cameras`: https://idtvision.com/products/cameras/ccm-series-cameras/