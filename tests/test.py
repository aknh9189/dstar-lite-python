import pytest
import dstar_lite
import numpy as np 

def setup_valid_dstar():
    np.random.seed(42)
    valid_map = np.random.rand(5000,5000) + 1
    dstar = dstar_lite.Dstar(valid_map)
    return dstar

def test_constructor():
    np.random.seed(42)
    # 3d map
    bad_map = np.random.rand(5000,5000,3)
    with pytest.raises(ValueError):
        dstar_lite.Dstar(bad_map)

    # wrong cost type
    bad_map = np.random.rand(5000,5000).astype(np.float32)
    with pytest.raises(TypeError):
        dstar_lite.Dstar(bad_map)
    
    # values below one in map
    bad_map = np.random.rand(5000,5000)
    bad_map[0,0] = 0.99
    with pytest.raises(RuntimeError):
        dstar_lite.Dstar(bad_map)

    # successful creation
    dstar = setup_valid_dstar()
    assert dstar is not None

def test_init():
    np.random.seed(42)
    # test valid init
    dstar = setup_valid_dstar()
    dstar.init(0,0,4999,4999)

    # test invalid init
    with pytest.raises(RuntimeError):
        dstar.init(-1,0,4999,4999)
    with pytest.raises(RuntimeError):
        dstar.init(1,0,5000,4999)

def test_updateStart():

    dstar = setup_valid_dstar()
    dstar.init(0,0,4999,4999)
    dstar.updateStart(0,0)
    dstar.updateStart(3232, 3232)

    with pytest.raises(RuntimeError):
        dstar.updateStart(-1,0)
    with pytest.raises(RuntimeError):
        dstar.updateStart(4999, 5000)

    # call without init
    dstar = setup_valid_dstar()
    with pytest.raises(RuntimeError):
        dstar.updateStart(0,0)

def test_replan():
    dstar = setup_valid_dstar()
    dstar.init(0,0,4999,4999)
    dstar.replan() 

    # call without init
    dstar = setup_valid_dstar()
    with pytest.raises(RuntimeError):
        dstar.replan()

    print("NEED TO ADD MORE TESTS TO REPLAN FUNCTION")