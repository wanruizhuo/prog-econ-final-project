.. _analysis:

***********************
Main model estimations
***********************

Documentation of the code in *src.analysis*. This is the core of the project.


Test_panel_model
=================

.. automodule:: src.analysis.test_panel_model
    :members:

I put this part into analysis for the reason that test_panel_model function is used to analyze the panel model function with real dataset. Moreover, it is also an analysis for observing whether the panel model function works properly when comparing its results to the true value.

Hausman
=======
.. automodule:: src.analysis.hausman
    :members:

Including this part is useful for obtaining the conclusion of Hausman Specification Test. By applying this part it is clear in this project to see whether the null hypothesis should be rejected or not. Hence, it is also clear to see GLS estimator is consistent or not and random effects model is suitable or not.