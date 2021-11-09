from predict import rank_img_combo
import pytest
import mock
import os
from unittest.mock import Mock
import glob


# def test_get_file_paths():
#     full_file_paths = ['2-DSC01120.ARW', '1-DSC01121.ARW', '1-DSC01122.ARW']
#     actual = predict.get_file_paths(r'C:\src\ninetest\arw-workflow', 'ARW')
#     expected = full_file_paths
#     assert actual == expected
#
#
# def test_get_undo_string():
#     actual = predict.powershell_undo_string(predict.get_file_paths(r'C:\src\ninetest\arw-workflow', 'ARW'))
#     expected = "'1-DSC01120.ARW', 'DSC01120.ARW', '2-DSC01121.ARW', 'DSC01121.ARW', '3-DSC01122.ARW', 'DSC01122.ARW'"
#     assert actual == expected

#
# def test_rank_img():
#     actual = rank_img(base_model_name='MobileNet',
#                       weights_aesthetic='weights_aesthetic.hdf5',
#                       weights_technical='weights_technical.hdf5',
#                       img_dir=r'C:\src\ninetest\arw-workflow',
#                       ext='ARW')
#     expected = ''
#     assert actual == expected
