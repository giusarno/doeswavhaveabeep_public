import torch
import logging

log = logging.getLogger(__name__)

def sub_1d_tensor(tensor,start,end):
    log.debug("sub_tensor shape {} start {} end {}".format(tensor.shape,start,end))
    res_tensor = tensor [start-1 : end]
    log.debug("sub_tensor new shape {}".format(res_tensor))
    return res_tensor

def sub_from_begin_1d_tensor(tensor,number):
    log.debug("sub_tensor shape {} start {} till the end".format(tensor.shape,number))
    res_tensor = tensor [number:]
    log.debug("sub_tensor new shape {}".format(res_tensor))
    return res_tensor

def sub_from_end_1d_tensor(tensor,number):
    log.debug("sub_tensor shape {} start {} till the end".format(tensor.shape,number))
    res_tensor = tensor [:-number]
    log.debug("sub_tensor new shape {}".format(res_tensor))
    return res_tensor

def sub_from_begin_2nd_tensor(tensor,number):
    log.debug("sub_tensor shape {} start {} till the end".format(tensor.shape,number))
    res_tensor = tensor [:,number:]
    log.debug("sub_tensor new shape {}".format(res_tensor))
    return res_tensor



