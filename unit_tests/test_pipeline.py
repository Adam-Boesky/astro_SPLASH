import numpy as np
import pkg_resources
import pytest
import sklearn

from SPLASH.pipeline import Splash_Pipeline

from .utils import is_close


def test_transform_photometry():
    pipeline = Splash_Pipeline(pipeline_version='full_band_no_photozs_zloss')
    test_photo = np.linspace(0.1, 1, num=18).reshape(1, -1)
    test_photo_errs = test_photo / 10

    for hp in (True, False):
        photo_trans, photo_trans_err = pipeline._transform_photometry(test_photo, test_photo_errs, just_grizy=False, host_prop=hp)
        photo_untrans, photo_untrans_err = pipeline._inverse_transform_photometry(photo_trans, photo_trans_err, just_grizy=False, host_prop=hp)

        assert is_close(photo_untrans, test_photo)
        assert is_close(photo_untrans_err, test_photo_errs)


def test_transform_properties():
    pipeline = Splash_Pipeline(pipeline_version='full_band_no_photozs_zloss')
    test_props = np.linspace(0.1, 1, num=3).reshape(1, -1)

    prop_trans = pipeline._inverse_transform_properties(test_props)[0]
    prop_untrans = pipeline._tranform_properties(prop_trans)

    assert is_close(prop_untrans, test_props)


def test_impute_photometry():
    pipeline = Splash_Pipeline(pipeline_version='full_band_no_photozs_zloss')
    test_photo = np.array([[np.nan, 0.2, 0.3, np.nan, 0.5]])
    test_photo_errs = test_photo / 10

    imputed_photo, imputed_photo_errs = pipeline._impute_photometry(test_photo, test_photo_errs)

    assert not np.isnan(imputed_photo).any()
    assert not np.isnan(imputed_photo_errs).any()


def test_transfer_domain():
    pipeline = Splash_Pipeline(pipeline_version='full_band_no_photozs_zloss')
    test_grizy = np.linspace(0.1, 0.5, num=5).reshape(1, -1)

    full_band = pipeline._transfer_domain(test_grizy)

    assert full_band.shape == (1, 18)


def test_predict_host_properties():
    pipeline = Splash_Pipeline(pipeline_version='full_band_no_photozs_zloss')
    test_grizy = np.linspace(0.1, 0.5, num=5).reshape(1, -1)
    test_grizy_err = test_grizy / 10

    host_props, host_props_err = pipeline.predict_host_properties(test_grizy, test_grizy_err)

    assert host_props.shape == (1, 3)
    assert host_props_err.shape == (1, 3)


def test_predict_classes():
    pipeline = Splash_Pipeline(pipeline_version='full_band_no_photozs_zloss')
    test_grizy = np.linspace(0.1, 0.5, num=5).reshape(1, -1)
    test_angular_sep = np.array([1.0]).reshape(1, -1)
    test_grizy_err = test_grizy / 10

    classes = pipeline.predict_classes(test_grizy, test_angular_sep, test_grizy_err)

    assert classes.shape == (1,)


def test_predict_probs():
    pipeline = Splash_Pipeline(pipeline_version='full_band_no_photozs_zloss')
    test_grizy = np.linspace(0.1, 0.5, num=5).reshape(1, -1)
    test_angular_sep = np.array([1.0]).reshape(1, -1)
    test_grizy_err = test_grizy / 10

    probs = pipeline.predict_probs(test_grizy, test_angular_sep, test_grizy_err)

    assert probs.shape == (1, 5)


def test_sklearn_versioning():
    pipeline = Splash_Pipeline(pipeline_version='full_band_no_photozs_zloss')
    
    if pkg_resources.parse_version(sklearn.__version__) >= pkg_resources.parse_version('1.3.0'):
        assert pipeline.sklearn_version == 'new_skl'
        assert pipeline.rf_fname == 'rf_classifier_new_version.pbz2'
    else:
        assert pipeline.sklearn_version == 'old_skl'
        assert pipeline.rf_fname == 'rf_classifier_old_version.pbz2'


def test_pipeline_init_invalid_version():
    with pytest.raises(ValueError, match="is not an option for your pipeline version"):
        Splash_Pipeline(pipeline_version='invalid_version')
