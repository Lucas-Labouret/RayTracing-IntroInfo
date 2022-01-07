import types
from random import randint
from math import sqrt
from csv import DictReader
from sys import argv, exit
global_dict = globals()
global_dict['randint'] = randint
global_dict['DictReader'] = DictReader
global_dict['sqrt'] = sqrt
global_dict['argv'] = argv
global_dict['exit'] = exit


sym = [b'\x41\x64\x64\x69\x74\x69\x6f\x6e\x20\x64\x65\x20\x64\x65\x75\x78\x20\x76\x65\x63\x74\x65\x75\x72\x73', b'\x76\x31', b'\x76\x32', b'\x3c\x73\x74\x72\x69\x6e\x67\x3e', b'\x61\x64\x64', b'\x44\x69\x66\x66\xc3\xa9\x72\x65\x6e\x63\x65\x20\x64\x65\x20\x64\x65\x75\x78\x20\x76\x65\x63\x74\x65\x75\x72\x73', b'\x73\x75\x62', b'\x4d\x75\x6c\x74\x69\x70\x6c\x69\x63\x61\x74\x69\x6f\x6e\x20\x70\x61\x72\x20\x75\x6e\x65\x20\x63\x6f\x6e\x73\x74\x61\x6e\x74\x65', b'\x6b', b'\x76', b'\x6d\x75\x6c', b'\x50\x72\x6f\x64\x75\x69\x74\x20\x73\x63\x61\x6c\x61\x69\x72\x65\x20\x64\x65\x20\x64\x65\x75\x78\x20\x76\x65\x63\x74\x65\x75\x72\x73', b'\x64\x6f\x74', b'\x4e\x6f\x72\x6d\x65\x20\x64\x27\x75\x6e\x20\x76\x65\x63\x74\x65\x75\x72\x20', b'\x73\x71\x72\x74', b'\x6e\x6f\x72\x6d', b'\x6e\x6f\x72\x6d\x61\x6c\x69\x7a\x65', b'\x46\x6f\x6e\x63\x74\x69\x6f\x6e\x20\x64\x65\x20\x74\x65\x73\x74\x20\x70\x6f\x75\x72\x20\x61\x64\x64', b'\x74\x65\x73\x74\x5f\x61\x64\x64', b'\x46\x6f\x6e\x63\x74\x69\x6f\x6e\x20\x64\x65\x20\x74\x65\x73\x74\x20\x70\x6f\x75\x72\x20\x73\x75\x62', b'\x74\x65\x73\x74\x5f\x73\x75\x62', b'\x46\x6f\x6e\x63\x74\x69\x6f\x6e\x20\x64\x65\x20\x74\x65\x73\x74\x20\x70\x6f\x75\x72\x20\x6d\x75\x6c', b'\x74\x65\x73\x74\x5f\x6d\x75\x6c', b'\x46\x6f\x6e\x63\x74\x69\x6f\x6e\x20\x64\x65\x20\x74\x65\x73\x74\x20\x70\x6f\x75\x72\x20\x64\x6f\x74', b'\x76\x33', b'\x74\x65\x73\x74\x5f\x64\x6f\x74', b'\x46\x6f\x6e\x63\x74\x69\x6f\x6e\x20\x64\x65\x20\x74\x65\x73\x74\x20\x70\x6f\x75\x72\x20\x6e\x6f\x72\x6d', b'\x74\x65\x73\x74\x5f\x6e\x6f\x72\x6d', b'\x46\x6f\x6e\x63\x74\x69\x6f\x6e\x20\x64\x65\x20\x74\x65\x73\x74\x20\x70\x6f\x75\x72\x20\x6e\x6f\x72\x6d\x61\x6c\x69\x7a\x65', b'\x74\x65\x73\x74\x5f\x6e\x6f\x72\x6d\x61\x6c\x69\x7a\x65', b'\x49\x6e\x69\x74\x69\x61\x6c\x69\x73\x65\x20\x75\x6e\x65\x20\x69\x6d\x61\x67\x65\x20\x64\x65\x20\x77\x20\x70\x69\x78\x65\x6c\x73\x20\x64\x65\x20\x6c\x61\x72\x67\x65\x20\x65\x74\x20\x68\x20\x70\x69\x78\x65\x6c\x20\x64\x65\x20\x68\x61\x75\x74', b'\x62\x79\x74\x65\x61\x72\x72\x61\x79', b'\x77', b'\x68', b'\x69\x6d\x67', b'\x69\x6e\x69\x74\x5f\x69\x6d\x61\x67\x65', b'\x4d\x65\x74\x20\x6c\x65\x20\x70\x69\x78\x65\x6c\x20\x61\x75\x20\x63\x6f\x6f\x72\x64\x6f\x6e\x6e\xc3\xa9\x65\x73\x20\x28\x78\x2c\x20\x79\x29\x20\xc3\xa0\x20\x6c\x61\x20\x63\x6f\x75\x6c\x65\x75\x72\x20\x63\x2e\x20\x43\x27\x65\x73\x74\x20\x75\x6e\x20\x65\x73\x74\x0a\x20\x20\x74\x72\x69\x70\x6c\x65\x74\x20\x28\x72\x2c\x20\x76\x2c\x20\x62\x29\x20\x64\x65\x20\x76\x61\x6c\x65\x75\x72\x73\x2e\x20\x4c\x65\x73\x20\x76\x61\x6c\x65\x75\x72\x73\x20\x73\x75\x70\xc3\xa9\x72\x69\x65\x75\x72\x65\x73\x20\xc3\xa0\x20\x31\x20\x28\x72\x65\x73\x70\x2e\x20\x69\x6e\x66\xc3\xa9\x72\x69\x65\x75\x72\x65\x73\x20\xc3\xa0\x0a\x20\x20\x30\x29\x20\x73\x6f\x6e\x74\x20\x6d\x69\x73\x65\x73\x20\xc3\xa0\x20\x31\x20\x28\x72\x65\x73\x70\x2e\x20\x30\x29\x2e\x0a\x20\x20', b'\x6d\x61\x78', b'\x6d\x69\x6e', b'\x69\x6e\x74', b'\x78', b'\x79', b'\x63', b'\x62\x75\x66\x66', b'\x69\x64\x78', b'\x73\x65\x74\x5f\x70\x69\x78\x65\x6c', b'\xc3\x89\x63\x72\x69\x74\x20\x6c\x27\x69\x6d\x61\x67\x65\x20\x69\x6d\x67\x20\x64\x61\x6e\x73\x20\x6c\x65\x20\x66\x69\x63\x68\x69\x65\x72\x20\x64\x6f\x6e\x74\x20\x6c\x65\x20\x63\x68\x65\x6d\x69\x6e\x20\x65\x73\x74\x20\x64\x6f\x6e\x6e\xc3\xa9\x2e\x20\x53\x69\x0a\x20\x20\x6c\x65\x20\x66\x69\x63\x68\x69\x65\x72\x20\x65\x78\x69\x73\x74\x65\x2c\x20\x69\x6c\x20\x65\x73\x74\x20\x73\x75\x70\x70\x72\x69\x6d\xc3\xa9\x2e\x20\x4c\x27\x69\x6d\x61\x67\x65\x20\x65\x73\x74\x20\x73\x74\x6f\x63\x6b\xc3\xa9\x65\x20\x61\x75\x20\x66\x6f\x72\x6d\x61\x74\x20\x50\x50\x4d', b'\x77\x62', b'\x6f\x70\x65\x6e', b'\x77\x72\x69\x74\x65', b'\x73\x74\x72', b'\x65\x6e\x63\x6f\x64\x65', b'\x63\x68\x65\x6d\x69\x6e', b'\x66', b'\x73\x61\x76\x65\x5f\x69\x6d\x61\x67\x65', b'\x62\x6c\x61\x63\x6b\x31\x30\x30\x2e\x70\x70\x6d', b'\x74\x65\x73\x74\x5f\x69\x6d\x67\x2e\x70\x70\x6d', b'\x72\x61\x6e\x67\x65', b'\x69\x6d\x31', b'\x69\x6d\x32', b'\x69', b'\x6a', b'\x63\x6f\x75\x6c\x65\x75\x72', b'\x74\x65\x73\x74\x5f\x69\x6d\x67', b'\x78\x6d\x69\x6e', b'\x78\x6d\x61\x78', b'\x79\x6d\x69\x6e', b'\x79\x6d\x61\x78', b'\x70\x78', b'\x70\x79', b'\x70\x69\x78\x65\x6c\x5f\x74\x6f\x5f\x70\x6f\x69\x6e\x74', b'\x72', b'\x64', b'\x76\x72\x61\x64', b'\x62', b'\x64\x65\x6c\x74\x61', b'\x73\x71\x64', b'\x74\x32', b'\x73\x70\x68\x65\x72\x65\x5f\x69\x6e\x74\x65\x72\x73\x65\x63\x74', b'\x69\x6e\x66', b'\x63\x65\x6e\x74\x65\x72', b'\x72\x61\x64\x69\x75\x73', b'\x66\x6c\x6f\x61\x74', b'\x6f\x62\x6a\x73', b'\x6f', b'\x6d\x69\x6e\x5f\x64\x69\x73\x74', b'\x6d\x69\x6e\x5f\x6f\x62\x6a', b'\x6f\x62\x6a', b'\x6e\x65\x61\x72\x73\x65\x74\x5f\x69\x6e\x74\x65\x72\x73\x65\x63\x74\x69\x6f\x6e', b'\x61\x6d\x62\x69\x61\x6e\x74', b'\x64\x69\x66\x66\x75\x73\x65', b'\x73\x68\x69\x6e\x69\x6e\x65\x73\x73', b'\x73\x70\x65\x63\x75\x6c\x61\x72', b'\x61\x62\x73', b'\x6e', b'\x6c', b'\x61', b'\x73', b'\x63\x6f\x6d\x70\x75\x74\x65\x5f\x63\x6f\x6c\x6f\x72', b'\x63\x61\x6d\x65\x72\x61', b'\x6c\x69\x67\x68\x74', b'\x70', b'\x76\x70', b'\x64\x69\x73\x74', b'\x78\x5f\x70\x6f\x69\x6e\x74', b'\x6f\x62\x73\x74\x61\x63\x6c\x65', b'\x64\x69\x73\x74\x5f\x6f\x62\x73\x74', b'\x74\x72\x61\x63\x65', b'\x2c', b'\x45\x72\x72\x65\x75\x72\x20\x64\x65\x20\x63\x68\x61\x72\x67\x65\x6d\x65\x6e\x74', b'\x73\x70\x6c\x69\x74', b'\x6c\x65\x6e', b'\x56\x61\x6c\x75\x65\x45\x72\x72\x6f\x72', b'\x66\x69\x65\x6c\x64\x73', b'\x72\x65\x61\x64\x5f\x76\x65\x63\x74\x6f\x72', b'\x43\x68\x61\x72\x67\x65\x20\x75\x6e\x20\x66\x69\x63\x68\x69\x65\x72\x20\x64\x65\x20\x64\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20\x64\x65\x20\x73\x63\xc3\xa8\x6e\x65\x2e\x20\x45\x6e\x20\x63\x61\x73\x20\x64\x27\x65\x72\x72\x65\x75\x72\x2c\x20\x6c\x61\x20\x66\x6f\x6e\x63\x74\x69\x6f\x6e\x0a\x20\x20\x20\x20\x20\x6c\xc3\xa8\x76\x65\x20\x75\x6e\x65\x20\x65\x78\x63\x65\x70\x74\x69\x6f\x6e\x20\x27\x45\x78\x63\x65\x70\x74\x69\x6f\x6e\x28\x22\x45\x72\x72\x65\x75\x72\x20\x64\x65\x20\x63\x68\x61\x72\x67\x65\x6d\x65\x6e\x74\x22\x29\x27\x0a\x20\x20', b'\x3b', b'\x72\x65\x66\x6c\x65\x63\x74\x69\x6f\x6e', b'\x72\x65\x61\x64\x6c\x69\x6e\x65', b'\x6c\x69\x73\x74', b'\x44\x69\x63\x74\x52\x65\x61\x64\x65\x72', b'\x6f\x62\x6a\x65\x63\x74\x73', b'\x6c\x6f\x61\x64\x5f\x73\x63\x65\x6e\x65', b'\x55\x73\x61\x67\x65\x3a\x20', b'\x20\x3c\x66\x69\x63\x68\x69\x65\x72\x2e\x73\x63\x65\x6e\x65\x3e', b'\x70\x72\x69\x6e\x74', b'\x61\x72\x67\x76', b'\x65\x78\x69\x74', b'\x75\x73\x61\x67\x65']
sym = [ s.decode() for s in sym ]
add = types.FunctionType(types.CodeType( 2 ,
    0 ,
    0 ,
    2 ,
    5 ,
    67 ,
    b'|\x00d\x01\x19\x00|\x01d\x01\x19\x00\x17\x00|\x00d\x02\x19\x00|\x01d\x02\x19\x00\x17\x00|\x00d\x03\x19\x00|\x01d\x03\x19\x00\x17\x00f\x03S\x00' ,
    (sym[0], 0, 1, 2) ,
    () ,
    (sym[1], sym[2]) ,
    sym[3] ,
    sym[4] ,
    8 ,
    b'\x00\x02' ,
    () ,
    () ), global_dict)

sub = types.FunctionType(types.CodeType( 2 ,
    0 ,
    0 ,
    2 ,
    5 ,
    67 ,
    b'|\x00d\x01\x19\x00|\x01d\x01\x19\x00\x18\x00|\x00d\x02\x19\x00|\x01d\x02\x19\x00\x18\x00|\x00d\x03\x19\x00|\x01d\x03\x19\x00\x18\x00f\x03S\x00' ,
    (sym[5], 0, 1, 2) ,
    () ,
    (sym[1], sym[2]) ,
    sym[3] ,
    sym[6] ,
    12 ,
    b'\x00\x02' ,
    () ,
    () ), global_dict)

mul = types.FunctionType(types.CodeType( 2 ,
    0 ,
    0 ,
    2 ,
    5 ,
    67 ,
    b'|\x00|\x01d\x01\x19\x00\x14\x00|\x00|\x01d\x02\x19\x00\x14\x00|\x00|\x01d\x03\x19\x00\x14\x00f\x03S\x00' ,
    (sym[7], 0, 1, 2) ,
    () ,
    (sym[8], sym[9]) ,
    sym[3] ,
    sym[10] ,
    16 ,
    b'\x00\x02' ,
    () ,
    () ), global_dict)

dot = types.FunctionType(types.CodeType( 2 ,
    0 ,
    0 ,
    2 ,
    4 ,
    67 ,
    b'|\x00d\x01\x19\x00|\x01d\x01\x19\x00\x14\x00|\x00d\x02\x19\x00|\x01d\x02\x19\x00\x14\x00\x17\x00|\x00d\x03\x19\x00|\x01d\x03\x19\x00\x14\x00\x17\x00S\x00' ,
    (sym[11], 0, 1, 2) ,
    () ,
    (sym[1], sym[2]) ,
    sym[3] ,
    sym[12] ,
    20 ,
    b'\x00\x02' ,
    () ,
    () ), global_dict)

norm = types.FunctionType(types.CodeType( 1 ,
    0 ,
    0 ,
    1 ,
    4 ,
    67 ,
    b't\x00t\x01|\x00|\x00\x83\x02\x83\x01S\x00' ,
    (sym[13], ) ,
    (sym[14], sym[12]) ,
    (sym[9], ) ,
    sym[3] ,
    sym[15] ,
    24 ,
    b'\x00\x02' ,
    () ,
    () ), global_dict)

normalize = types.FunctionType(types.CodeType( 1 ,
    0 ,
    0 ,
    1 ,
    4 ,
    67 ,
    b't\x00d\x01t\x01|\x00\x83\x01\x1b\x00|\x00\x83\x02S\x00' ,
    (None, 1) ,
    (sym[10], sym[15]) ,
    (sym[9], ) ,
    sym[3] ,
    sym[16] ,
    28 ,
    b'\x00\x01' ,
    () ,
    () ), global_dict)

test_add = types.FunctionType(types.CodeType( 0 ,
    0 ,
    0 ,
    2 ,
    4 ,
    67 ,
    b'd\x01}\x00d\x02}\x01t\x00|\x00|\x01\x83\x02t\x00|\x01|\x00\x83\x02k\x02s J\x00\x82\x01t\x00|\x00|\x01\x83\x02d\x03k\x02s2J\x00\x82\x01d\x04S\x00' ,
    (sym[17], (1, 1, 1), (1, 2, 3), (2, 3, 4), None) ,
    (sym[4], ) ,
    (sym[1], sym[2]) ,
    sym[3] ,
    sym[18] ,
    31 ,
    b'\x00\x02\x04\x01\x04\x01\x18\x01' ,
    () ,
    () ), global_dict)

test_sub = types.FunctionType(types.CodeType( 0 ,
    0 ,
    0 ,
    2 ,
    3 ,
    67 ,
    b'd\x01}\x00d\x02}\x01t\x00|\x00|\x01\x83\x02d\x03k\x02s\x1aJ\x00\x82\x01t\x00|\x00d\x04\x83\x02|\x00k\x02s,J\x00\x82\x01d\x05S\x00' ,
    (sym[19], (1, 1, 1), (1, 2, 3), (0, -1, -2), (0, 0, 0), None) ,
    (sym[6], ) ,
    (sym[1], sym[2]) ,
    sym[3] ,
    sym[20] ,
    40 ,
    b'\x00\x02\x04\x01\x04\x01\x12\x01' ,
    () ,
    () ), global_dict)

test_mul = types.FunctionType(types.CodeType( 0 ,
    0 ,
    0 ,
    1 ,
    3 ,
    67 ,
    b'd\x01}\x00t\x00d\x02|\x00\x83\x02d\x03k\x02s\x16J\x00\x82\x01t\x00d\x04|\x00\x83\x02d\x05k\x02s(J\x00\x82\x01d\x06S\x00' ,
    (sym[21], (1, 2, 3), 3, (3, 6, 9), -3, (-3, -6, -9), None) ,
    (sym[10], ) ,
    (sym[1], ) ,
    sym[3] ,
    sym[22] ,
    49 ,
    b'\x00\x02\x04\x01\x12\x01' ,
    () ,
    () ), global_dict)

test_dot = types.FunctionType(types.CodeType( 0 ,
    0 ,
    0 ,
    3 ,
    3 ,
    67 ,
    b'd\x01}\x00d\x02}\x01d\x03}\x02t\x00|\x00|\x01\x83\x02d\x04k\x02s\x1eJ\x00\x82\x01t\x00|\x02|\x01\x83\x02d\x04k\x02s0J\x00\x82\x01d\x05S\x00' ,
    (sym[23], (1, 0, 0), (0, 0, 1), (1, 1, 0), 0, None) ,
    (sym[12], ) ,
    (sym[1], sym[2], sym[24]) ,
    sym[3] ,
    sym[25] ,
    57 ,
    b'\x00\x02\x04\x01\x04\x01\x04\x01\x12\x01' ,
    () ,
    () ), global_dict)

test_norm = types.FunctionType(types.CodeType( 0 ,
    0 ,
    0 ,
    0 ,
    3 ,
    67 ,
    b't\x00d\x01\x83\x01d\x02k\x02s\x10J\x00\x82\x01t\x00d\x03\x83\x01t\x01d\x04\x83\x01k\x02s$J\x00\x82\x01d\x05S\x00' ,
    (sym[26], (1, 0, 0), 1, (1, 1, 1), 3, None) ,
    (sym[15], sym[14]) ,
    () ,
    sym[3] ,
    sym[27] ,
    67 ,
    b'\x00\x02\x10\x01' ,
    () ,
    () ), global_dict)

test_normalize = types.FunctionType(types.CodeType( 0 ,
    0 ,
    0 ,
    0 ,
    3 ,
    67 ,
    b't\x00t\x01d\x01\x83\x01\x83\x01d\x02k\x02s\x14J\x00\x82\x01t\x00t\x01d\x03\x83\x01\x83\x01d\x02k\x02s(J\x00\x82\x01d\x04S\x00' ,
    (sym[28], (2, 0, 0), 1, (4, 0, 0), None) ,
    (sym[15], sym[16]) ,
    () ,
    sym[3] ,
    sym[29] ,
    74 ,
    b'\x00\x02\x14\x01' ,
    () ,
    () ), global_dict)

init_image = types.FunctionType(types.CodeType( 2 ,
    0 ,
    0 ,
    3 ,
    3 ,
    67 ,
    b't\x00|\x00|\x01\x14\x00d\x01\x14\x00\x83\x01}\x02|\x02|\x00|\x01f\x03S\x00' ,
    (sym[30], 3) ,
    (sym[31], ) ,
    (sym[32], sym[33], sym[34]) ,
    sym[3] ,
    sym[35] ,
    89 ,
    b'\x00\x02\x10\x01' ,
    () ,
    () ), global_dict)

set_pixel = types.FunctionType(types.CodeType( 4 ,
    0 ,
    0 ,
    8 ,
    7 ,
    67 ,
    b'|\x00\\\x03}\x04}\x05}\x06|\x02d\x01\x14\x00|\x05\x14\x00d\x01|\x01\x14\x00\x17\x00}\x07t\x00d\x02t\x01d\x03t\x02|\x03d\x02\x19\x00d\x03\x14\x00\x83\x01\x83\x02\x83\x02|\x04|\x07<\x00t\x00d\x02t\x01d\x03t\x02|\x03d\x04\x19\x00d\x03\x14\x00\x83\x01\x83\x02\x83\x02|\x04|\x07d\x04\x17\x00<\x00t\x00d\x02t\x01d\x03t\x02|\x03d\x05\x19\x00d\x03\x14\x00\x83\x01\x83\x02\x83\x02|\x04|\x07d\x05\x17\x00<\x00d\x06S\x00' ,
    (sym[36], 3, 0, 255, 1, 2, None) ,
    (sym[37], sym[38], sym[39]) ,
    (sym[34], sym[40], sym[41], sym[42], sym[43], sym[32], sym[33], sym[44]) ,
    sym[3] ,
    sym[45] ,
    94 ,
    b'\x00\x05\n\x01\x14\x01 \x01$\x01' ,
    () ,
    () ), global_dict)

save_image = types.FunctionType(types.CodeType( 2 ,
    0 ,
    0 ,
    6 ,
    8 ,
    67 ,
    b'|\x01\\\x03}\x02}\x03}\x04t\x00|\x00d\x01\x83\x02\x8f\\}\x05|\x05\xa0\x01d\x02\xa1\x01\x01\x00|\x05\xa0\x01t\x02|\x03\x83\x01\xa0\x03\xa1\x00\xa1\x01\x01\x00|\x05\xa0\x01d\x03\xa1\x01\x01\x00|\x05\xa0\x01t\x02|\x04\x83\x01\xa0\x03\xa1\x00\xa1\x01\x01\x00|\x05\xa0\x01d\x04\xa1\x01\x01\x00|\x05\xa0\x01|\x02\xa1\x01\x01\x00W\x00d\x05\x04\x00\x04\x00\x83\x03\x01\x00n\x101\x00sv0\x00\x01\x00\x01\x00\x01\x00Y\x00\x01\x00d\x05S\x00' ,
    (sym[46], sym[47], b'P6\n', b' ', b'\n255\n', None) ,
    (sym[48], sym[49], sym[50], sym[51]) ,
    (sym[52], sym[34], sym[43], sym[32], sym[33], sym[53]) ,
    sym[3] ,
    sym[54] ,
    105 ,
    b'\x00\x03\n\x01\x0c\x01\n\x01\x12\x01\n\x01\x12\x01\n\x01' ,
    () ,
    () ), global_dict)

test_img = types.FunctionType(types.CodeType( 0 ,
    0 ,
    0 ,
    5 ,
    7 ,
    67 ,
    b't\x00d\x01d\x01\x83\x02}\x00t\x01d\x02|\x00\x83\x02\x01\x00t\x00d\x03d\x03\x83\x02}\x01t\x02d\x03\x83\x01D\x00]B}\x02t\x02d\x03\x83\x01D\x00]4}\x03|\x02d\x04\x1a\x00d\x05\x16\x00|\x03d\x04\x1a\x00d\x05\x16\x00k\x02rTd\x06}\x04n\x04d\x07}\x04t\x03|\x01|\x02|\x03|\x04\x83\x04\x01\x00q2q&t\x01d\x08|\x01\x83\x02\x01\x00d\x00S\x00' ,
    (None, 100, sym[55], 200, 10, 2, (1, 0, 0), (0, 1, 1), sym[56]) ,
    (sym[35], sym[54], sym[57], sym[45]) ,
    (sym[58], sym[59], sym[60], sym[61], sym[62]) ,
    sym[3] ,
    sym[63] ,
    117 ,
    b'\x00\x01\n\x01\n\x01\n\x02\x0c\x01\x0c\x01\x18\x01\x06\x02\x04\x01\x12\x01' ,
    () ,
    () ), global_dict)

pixel_to_point = types.FunctionType(types.CodeType( 8 ,
    0 ,
    0 ,
    8 ,
    3 ,
    67 ,
    b'|\x03|\x02\x18\x00|\x00\x1b\x00|\x06\x14\x00|\x02\x17\x00|\x05|\x04\x18\x00|\x01\x1b\x00|\x07\x14\x00|\x04\x17\x00f\x02S\x00' ,
    (None, ) ,
    () ,
    (sym[32], sym[33], sym[64], sym[65], sym[66], sym[67], sym[68], sym[69]) ,
    sym[3] ,
    sym[70] ,
    135 ,
    b'\x00\x01' ,
    () ,
    () ), global_dict)

sphere_intersect = types.FunctionType(types.CodeType( 4 ,
    0 ,
    0 ,
    9 ,
    4 ,
    67 ,
    b't\x00|\x02|\x00\x83\x02}\x04d\x01t\x01|\x03|\x04\x83\x02\x14\x00}\x05t\x02|\x04\x83\x01d\x01\x13\x00|\x01|\x01\x14\x00\x18\x00}\x00|\x05|\x05\x14\x00d\x02|\x00\x14\x00\x18\x00}\x06|\x06d\x03k\x04rft\x03|\x06\x83\x01}\x07|\x05\x0b\x00|\x07\x18\x00}\x08|\x08d\x03k\x04rfd\x04|\x08\x14\x00S\x00d\x00S\x00' ,
    (None, 2, 4, 0, 0.5) ,
    (sym[6], sym[12], sym[15], sym[14]) ,
    (sym[42], sym[71], sym[9], sym[72], sym[73], sym[74], sym[75], sym[76], sym[77]) ,
    sym[3] ,
    sym[78] ,
    138 ,
    b'\x00\x01\n\x01\x0e\x01\x14\x01\x10\x01\x08\x01\x08\x01\n\x01\x08\x01\x08\x01' ,
    () ,
    () ), global_dict)

nearset_intersection = types.FunctionType(types.CodeType( 3 ,
    0 ,
    0 ,
    7 ,
    6 ,
    67 ,
    b't\x00d\x01\x83\x01}\x03d\x00}\x04|\x00D\x00]2}\x05t\x01|\x05d\x02\x19\x00|\x05d\x03\x19\x00|\x01|\x02\x83\x04}\x06|\x06d\x00k\x03r\x10|\x06|\x03k\x00r\x10|\x06}\x03|\x05}\x04q\x10|\x04|\x03f\x02S\x00' ,
    (None, sym[79], sym[80], sym[81]) ,
    (sym[82], sym[78]) ,
    (sym[83], sym[84], sym[72], sym[85], sym[86], sym[87], sym[8]) ,
    sym[3] ,
    sym[88] ,
    151 ,
    b'\x00\x02\x08\x01\x04\x01\x08\x01\x16\x01\x10\x01\x04\x01\x06\x02' ,
    () ,
    () ), global_dict)

compute_color = types.FunctionType(types.CodeType( 4 ,
    0 ,
    0 ,
    7 ,
    8 ,
    67 ,
    b'|\x00d\x01\x19\x00}\x04t\x00t\x01|\x03|\x02\x83\x02|\x00d\x02\x19\x00\x83\x02}\x05t\x00t\x02t\x01|\x02t\x03t\x04|\x03|\x01\x83\x02\x83\x01\x83\x02\x83\x01|\x00d\x03\x19\x00d\x04\x1b\x00\x13\x00|\x00d\x05\x19\x00\x83\x02}\x06t\x04|\x04t\x04|\x05|\x06\x83\x02\x83\x02S\x00' ,
    (None, sym[89], sym[90], sym[91], 4, sym[92]) ,
    (sym[10], sym[12], sym[93], sym[16], sym[4]) ,
    (sym[87], sym[9], sym[94], sym[95], sym[96], sym[72], sym[97]) ,
    sym[3] ,
    sym[98] ,
    163 ,
    b'\x00\x01\x08\x01\x14\x01.\x01' ,
    () ,
    () ), global_dict)

trace = types.FunctionType(types.CodeType( 9 ,
    0 ,
    0 ,
    25 ,
    11 ,
    67 ,
    b't\x00|\x00|\x01\x83\x02}\tt\x01|\x01\x83\x01D\x00]\xe2}\nt\x01|\x00\x83\x01D\x00]\xd4}\x0bt\x02|\x00|\x01|\x02|\x03|\x04|\x05|\x0b|\n\x83\x08\\\x02}\x0c}\r|\x0c|\rd\x01f\x03}\x0et\x03|\x0e|\x06\x83\x02}\x0ft\x04|\x0f\x83\x01}\x10t\x05|\x08|\x06|\x10\x83\x03\\\x02}\x11}\x12|\x11d\x00u\x00rvd\x02}\x13nft\x06|\x06t\x07|\x12|\x10\x83\x02\x83\x02}\x14t\x04t\x03|\x07|\x14\x83\x02\x83\x01}\x15t\x05|\x08|\x14|\x15\x83\x03\\\x02}\x16}\x17|\x17t\x08t\x03|\x07|\x14\x83\x02\x83\x01k\x00r\xbcd\x02}\x13n t\x04t\x03|\x14|\x11d\x03\x19\x00\x83\x02\x83\x01}\x18t\t|\x11|\x06|\x18|\x15\x83\x04}\x13t\n|\t|\x0b|\x01|\n\x18\x00d\x04\x18\x00|\x13\x83\x04\x01\x00q\x1eq\x12|\tS\x00' ,
    (None, 0, (0, 0, 0), sym[80], 1) ,
    (sym[35], sym[57], sym[70], sym[6], sym[16], sym[88], sym[4], sym[10], sym[15], sym[98], sym[45]) ,
    (sym[32], sym[33], sym[64], sym[65], sym[66], sym[67], sym[99], sym[100], sym[83], sym[34], sym[69], sym[68], sym[40], sym[41], sym[101], sym[102], sym[72], sym[87], sym[103], sym[62], sym[104], sym[95], sym[105], sym[106], sym[94]) ,
    sym[3] ,
    sym[107] ,
    169 ,
    b'\x00\x02\n\x02\x0c\x01\x0c\x01\x1a\x01\n\x01\n\x01\x08\x02\x10\x01\x08\x01\x06\x02\x10\x01\x0e\x01\x10\x01\x12\x01\x06\x02\x12\x01\x0e\x02\x1a\x02' ,
    () ,
    () ), global_dict)

read_vector = types.FunctionType(types.CodeType( 1 ,
    0 ,
    0 ,
    3 ,
    4 ,
    67 ,
    b'|\x00\xa0\x00d\x01\xa1\x01}\x01t\x01|\x01\x83\x01d\x02k\x03r\x1et\x02d\x03\x83\x01\x82\x01t\x03d\x02\x83\x01D\x00]\x14}\x02t\x04|\x01|\x02\x19\x00\x83\x01|\x01|\x02<\x00q&|\x01S\x00' ,
    (None, sym[108], 3, sym[109]) ,
    (sym[110], sym[111], sym[112], sym[57], sym[82]) ,
    (sym[97], sym[113], sym[60]) ,
    sym[3] ,
    sym[114] ,
    202 ,
    b'\x00\x01\n\x01\x0c\x01\x08\x01\x0c\x01\x12\x01' ,
    () ,
    () ), global_dict)

load_scene = types.FunctionType(types.CodeType( 1 ,
    0 ,
    0 ,
    12 ,
    9 ,
    67 ,
    b'\x90\x01zHt\x00|\x00d\x01\x83\x02\x90\x01\x8f\x12}\x01t\x01|\x01\xa0\x02\xa1\x00\x83\x01}\x02t\x01|\x01\xa0\x02\xa1\x00\x83\x01}\x03t\x03|\x01\xa0\x02\xa1\x00\x83\x01}\x04t\x03|\x01\xa0\x02\xa1\x00\x83\x01}\x05t\x03|\x01\xa0\x02\xa1\x00\x83\x01}\x06t\x03|\x01\xa0\x02\xa1\x00\x83\x01}\x07t\x04|\x01\xa0\x02\xa1\x00\x83\x01}\x08t\x04|\x01\xa0\x02\xa1\x00\x83\x01}\tt\x05t\x06|\x01d\x02d\x03\x8d\x02\x83\x01}\n|\nD\x00]\x8c}\x0bt\x04|\x0bd\x04\x19\x00\x83\x01|\x0bd\x04<\x00t\x03|\x0bd\x05\x19\x00\x83\x01|\x0bd\x05<\x00t\x04|\x0bd\x06\x19\x00\x83\x01|\x0bd\x06<\x00t\x04|\x0bd\x07\x19\x00\x83\x01|\x0bd\x07<\x00t\x04|\x0bd\x08\x19\x00\x83\x01|\x0bd\x08<\x00t\x07d\tt\x08d\nt\x03|\x0bd\x0b\x19\x00\x83\x01\x83\x02\x83\x02|\x0bd\x0b<\x00t\x07d\x0ct\x08d\nt\x03|\x0bd\r\x19\x00\x83\x01\x83\x02\x83\x02|\x0bd\r<\x00q\x86W\x00d\x0e\x04\x00\x04\x00\x83\x03\x01\x00n\x121\x00\x90\x01s*0\x00\x01\x00\x01\x00\x01\x00Y\x00\x01\x00|\x02|\x03|\x04|\x05|\x06|\x07|\x08|\t|\nf\tW\x00S\x00\x01\x00\x01\x00\x01\x00t\td\x0f\x83\x01\x82\x01Y\x00n\x020\x00d\x0eS\x00' ,
    (sym[115], sym[71], sym[116], ('delimiter',), sym[80], sym[81], sym[89], sym[90], sym[92], 100, 0, sym[91], 1, sym[117], None, sym[109]) ,
    (sym[48], sym[39], sym[118], sym[82], sym[114], sym[119], sym[120], sym[38], sym[37], sym[112]) ,
    (sym[52], sym[53], sym[32], sym[33], sym[64], sym[65], sym[66], sym[67], sym[99], sym[100], sym[121], sym[87]) ,
    sym[3] ,
    sym[122] ,
    210 ,
    b'\x00\x04\x04\x01\x0e\x01\x0c\x01\x0c\x01\x0c\x01\x0c\x01\x0c\x01\x0c\x01\x0c\x01\x0c\x01\x10\x01\x08\x01\x10\x01\x10\x01\x10\x01\x10\x01\x10\x01\x1c\x01>\x01\x18\x01\x06\x01' ,
    () ,
    () ), global_dict)

usage = types.FunctionType(types.CodeType( 0 ,
    0 ,
    0 ,
    0 ,
    4 ,
    67 ,
    b't\x00d\x01t\x01d\x02\x19\x00\x9b\x00d\x03\x9d\x03\x83\x01\x01\x00t\x02d\x04\x83\x01\x01\x00d\x00S\x00' ,
    (None, sym[123], 0, sym[124], 1) ,
    (sym[125], sym[126], sym[127]) ,
    () ,
    sym[3] ,
    sym[128] ,
    237 ,
    b'\x00\x01\x14\x01' ,
    () ,
    () ), global_dict)

