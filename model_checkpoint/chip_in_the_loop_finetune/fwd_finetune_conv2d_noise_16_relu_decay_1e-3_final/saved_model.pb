��*
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-0-ge5bf8de8��&
�
activation_quant_14/reluxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameactivation_quant_14/relux

-activation_quant_14/relux/Read/ReadVariableOpReadVariableOpactivation_quant_14/relux*
_output_shapes
: *
dtype0
�
conv2d_noise_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameconv2d_noise_17/kernel
�
*conv2d_noise_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_noise_17/kernel*&
_output_shapes
:@@*
dtype0
�
conv2d_noise_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv2d_noise_17/bias
y
(conv2d_noise_17/bias/Read/ReadVariableOpReadVariableOpconv2d_noise_17/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_15/gamma
�
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_15/beta
�
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_15/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_15/moving_mean
�
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_15/moving_variance
�
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes
:@*
dtype0
�
activation_quant_15/reluxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameactivation_quant_15/relux

-activation_quant_15/relux/Read/ReadVariableOpReadVariableOpactivation_quant_15/relux*
_output_shapes
: *
dtype0
�
conv2d_noise_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameconv2d_noise_18/kernel
�
*conv2d_noise_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_noise_18/kernel*&
_output_shapes
:@@*
dtype0
�
conv2d_noise_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv2d_noise_18/bias
y
(conv2d_noise_18/bias/Read/ReadVariableOpReadVariableOpconv2d_noise_18/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_16/gamma
�
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_16/beta
�
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_16/moving_mean
�
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_16/moving_variance
�
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:@*
dtype0
�
activation_quant_16/reluxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameactivation_quant_16/relux

-activation_quant_16/relux/Read/ReadVariableOpReadVariableOpactivation_quant_16/relux*
_output_shapes
: *
dtype0
�
conv2d_noise_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameconv2d_noise_19/kernel
�
*conv2d_noise_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_noise_19/kernel*&
_output_shapes
:@@*
dtype0
�
conv2d_noise_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv2d_noise_19/bias
y
(conv2d_noise_19/bias/Read/ReadVariableOpReadVariableOpconv2d_noise_19/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_17/gamma
�
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_17/beta
�
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_17/moving_mean
�
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_17/moving_variance
�
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:@*
dtype0
�
activation_quant_17/reluxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameactivation_quant_17/relux

-activation_quant_17/relux/Read/ReadVariableOpReadVariableOpactivation_quant_17/relux*
_output_shapes
: *
dtype0
�
conv2d_noise_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameconv2d_noise_20/kernel
�
*conv2d_noise_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_noise_20/kernel*&
_output_shapes
:@@*
dtype0
�
conv2d_noise_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv2d_noise_20/bias
y
(conv2d_noise_20/bias/Read/ReadVariableOpReadVariableOpconv2d_noise_20/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_18/gamma
�
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_18/beta
�
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_18/moving_mean
�
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_18/moving_variance
�
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes
:@*
dtype0
�
activation_quant_18/reluxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameactivation_quant_18/relux

-activation_quant_18/relux/Read/ReadVariableOpReadVariableOpactivation_quant_18/relux*
_output_shapes
: *
dtype0
�
dense_noise/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*#
shared_namedense_noise/kernel
y
&dense_noise/kernel/Read/ReadVariableOpReadVariableOpdense_noise/kernel*
_output_shapes

:@
*
dtype0
x
dense_noise/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_noise/bias
q
$dense_noise/bias/Read/ReadVariableOpReadVariableOpdense_noise/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
 Adam/activation_quant_14/relux/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_14/relux/m
�
4Adam/activation_quant_14/relux/m/Read/ReadVariableOpReadVariableOp Adam/activation_quant_14/relux/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_17/kernel/m
�
1Adam/conv2d_noise_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_17/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_17/bias/m
�
/Adam/conv2d_noise_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_17/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_15/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_15/gamma/m
�
7Adam/batch_normalization_15/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_15/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_15/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_15/beta/m
�
6Adam/batch_normalization_15/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_15/beta/m*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_15/relux/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_15/relux/m
�
4Adam/activation_quant_15/relux/m/Read/ReadVariableOpReadVariableOp Adam/activation_quant_15/relux/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_18/kernel/m
�
1Adam/conv2d_noise_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_18/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_18/bias/m
�
/Adam/conv2d_noise_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_18/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_16/gamma/m
�
7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_16/beta/m
�
6Adam/batch_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/m*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_16/relux/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_16/relux/m
�
4Adam/activation_quant_16/relux/m/Read/ReadVariableOpReadVariableOp Adam/activation_quant_16/relux/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_19/kernel/m
�
1Adam/conv2d_noise_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_19/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_19/bias/m
�
/Adam/conv2d_noise_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_19/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_17/gamma/m
�
7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_17/beta/m
�
6Adam/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/m*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_17/relux/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_17/relux/m
�
4Adam/activation_quant_17/relux/m/Read/ReadVariableOpReadVariableOp Adam/activation_quant_17/relux/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_20/kernel/m
�
1Adam/conv2d_noise_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_20/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_20/bias/m
�
/Adam/conv2d_noise_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_20/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_18/gamma/m
�
7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_18/beta/m
�
6Adam/batch_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/m*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_18/relux/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_18/relux/m
�
4Adam/activation_quant_18/relux/m/Read/ReadVariableOpReadVariableOp Adam/activation_quant_18/relux/m*
_output_shapes
: *
dtype0
�
Adam/dense_noise/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
**
shared_nameAdam/dense_noise/kernel/m
�
-Adam/dense_noise/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_noise/kernel/m*
_output_shapes

:@
*
dtype0
�
Adam/dense_noise/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_noise/bias/m

+Adam/dense_noise/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_noise/bias/m*
_output_shapes
:
*
dtype0
�
 Adam/activation_quant_14/relux/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_14/relux/v
�
4Adam/activation_quant_14/relux/v/Read/ReadVariableOpReadVariableOp Adam/activation_quant_14/relux/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_17/kernel/v
�
1Adam/conv2d_noise_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_17/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_17/bias/v
�
/Adam/conv2d_noise_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_17/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_15/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_15/gamma/v
�
7Adam/batch_normalization_15/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_15/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_15/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_15/beta/v
�
6Adam/batch_normalization_15/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_15/beta/v*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_15/relux/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_15/relux/v
�
4Adam/activation_quant_15/relux/v/Read/ReadVariableOpReadVariableOp Adam/activation_quant_15/relux/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_18/kernel/v
�
1Adam/conv2d_noise_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_18/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_18/bias/v
�
/Adam/conv2d_noise_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_18/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_16/gamma/v
�
7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_16/beta/v
�
6Adam/batch_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/v*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_16/relux/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_16/relux/v
�
4Adam/activation_quant_16/relux/v/Read/ReadVariableOpReadVariableOp Adam/activation_quant_16/relux/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_19/kernel/v
�
1Adam/conv2d_noise_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_19/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_19/bias/v
�
/Adam/conv2d_noise_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_19/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_17/gamma/v
�
7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_17/beta/v
�
6Adam/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/v*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_17/relux/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_17/relux/v
�
4Adam/activation_quant_17/relux/v/Read/ReadVariableOpReadVariableOp Adam/activation_quant_17/relux/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_20/kernel/v
�
1Adam/conv2d_noise_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_20/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_20/bias/v
�
/Adam/conv2d_noise_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_20/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_18/gamma/v
�
7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_18/beta/v
�
6Adam/batch_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/v*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_18/relux/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_18/relux/v
�
4Adam/activation_quant_18/relux/v/Read/ReadVariableOpReadVariableOp Adam/activation_quant_18/relux/v*
_output_shapes
: *
dtype0
�
Adam/dense_noise/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
**
shared_nameAdam/dense_noise/kernel/v
�
-Adam/dense_noise/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_noise/kernel/v*
_output_shapes

:@
*
dtype0
�
Adam/dense_noise/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_noise/bias/v

+Adam/dense_noise/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_noise/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
ו
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer-20
layer_with_weights-12
layer-21
layer_with_weights-13
layer-22
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
 
R
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
R
&	variables
'regularization_losses
(trainable_variables
)	keras_api
]
	*relux
+	variables
,regularization_losses
-trainable_variables
.	keras_api
h

/kernel
0bias
1	variables
2regularization_losses
3trainable_variables
4	keras_api
�
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;regularization_losses
<trainable_variables
=	keras_api
]
	>relux
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
h

Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
�
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
R
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
]
	Vrelux
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
h

[kernel
\bias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
�
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gregularization_losses
htrainable_variables
i	keras_api
]
	jrelux
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
h

okernel
pbias
q	variables
rregularization_losses
strainable_variables
t	keras_api
�
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{regularization_losses
|trainable_variables
}	keras_api
T
~	variables
regularization_losses
�trainable_variables
�	keras_api
V
�	variables
�regularization_losses
�trainable_variables
�	keras_api
V
�	variables
�regularization_losses
�trainable_variables
�	keras_api
b

�relux
�	variables
�regularization_losses
�trainable_variables
�	keras_api
n
�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate*m�/m�0m�6m�7m�>m�Cm�Dm�Jm�Km�Vm�[m�\m�bm�cm�jm�om�pm�vm�wm�	�m�	�m�	�m�*v�/v�0v�6v�7v�>v�Cv�Dv�Jv�Kv�Vv�[v�\v�bv�cv�jv�ov�pv�vv�wv�	�v�	�v�	�v�
�
*0
/1
02
63
74
85
96
>7
C8
D9
J10
K11
L12
M13
V14
[15
\16
b17
c18
d19
e20
j21
o22
p23
v24
w25
x26
y27
�28
�29
�30
 
�
*0
/1
02
63
74
>5
C6
D7
J8
K9
V10
[11
\12
b13
c14
j15
o16
p17
v18
w19
�20
�21
�22
�
�metrics
	variables
�layers
regularization_losses
trainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
 
 
 
�
�metrics
	variables
�layers
regularization_losses
 trainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
 
 
�
�metrics
"	variables
�layers
#regularization_losses
$trainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
 
 
�
�metrics
&	variables
�layers
'regularization_losses
(trainable_variables
 �layer_regularization_losses
�non_trainable_variables
db
VARIABLE_VALUEactivation_quant_14/relux5layer_with_weights-0/relux/.ATTRIBUTES/VARIABLE_VALUE

*0
 

*0
�
�metrics
+	variables
�layers
,regularization_losses
-trainable_variables
 �layer_regularization_losses
�non_trainable_variables
b`
VARIABLE_VALUEconv2d_noise_17/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_noise_17/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01
 

/0
01
�
�metrics
1	variables
�layers
2regularization_losses
3trainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_15/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_15/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_15/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_15/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

60
71
82
93
 

60
71
�
�metrics
:	variables
�layers
;regularization_losses
<trainable_variables
 �layer_regularization_losses
�non_trainable_variables
db
VARIABLE_VALUEactivation_quant_15/relux5layer_with_weights-3/relux/.ATTRIBUTES/VARIABLE_VALUE

>0
 

>0
�
�metrics
?	variables
�layers
@regularization_losses
Atrainable_variables
 �layer_regularization_losses
�non_trainable_variables
b`
VARIABLE_VALUEconv2d_noise_18/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_noise_18/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
�
�metrics
E	variables
�layers
Fregularization_losses
Gtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
L2
M3
 

J0
K1
�
�metrics
N	variables
�layers
Oregularization_losses
Ptrainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
 
 
�
�metrics
R	variables
�layers
Sregularization_losses
Ttrainable_variables
 �layer_regularization_losses
�non_trainable_variables
db
VARIABLE_VALUEactivation_quant_16/relux5layer_with_weights-6/relux/.ATTRIBUTES/VARIABLE_VALUE

V0
 

V0
�
�metrics
W	variables
�layers
Xregularization_losses
Ytrainable_variables
 �layer_regularization_losses
�non_trainable_variables
b`
VARIABLE_VALUEconv2d_noise_19/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_noise_19/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
 

[0
\1
�
�metrics
]	variables
�layers
^regularization_losses
_trainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

b0
c1
d2
e3
 

b0
c1
�
�metrics
f	variables
�layers
gregularization_losses
htrainable_variables
 �layer_regularization_losses
�non_trainable_variables
db
VARIABLE_VALUEactivation_quant_17/relux5layer_with_weights-9/relux/.ATTRIBUTES/VARIABLE_VALUE

j0
 

j0
�
�metrics
k	variables
�layers
lregularization_losses
mtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
ca
VARIABLE_VALUEconv2d_noise_20/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_noise_20/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

o0
p1
 

o0
p1
�
�metrics
q	variables
�layers
rregularization_losses
strainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_18/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_18/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_18/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_18/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
x2
y3
 

v0
w1
�
�metrics
z	variables
�layers
{regularization_losses
|trainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
 
 
�
�metrics
~	variables
�layers
regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
 
 
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
 
 
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
ec
VARIABLE_VALUEactivation_quant_18/relux6layer_with_weights-12/relux/.ATTRIBUTES/VARIABLE_VALUE

�0
 

�0
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
_]
VARIABLE_VALUEdense_noise/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdense_noise/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

�0
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
 
8
80
91
L2
M3
d4
e5
x6
y7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

80
91
 
 
 
 
 
 
 
 
 
 
 

L0
M1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

d0
e1
 
 
 
 
 
 
 
 
 
 
 

x0
y1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


�total

�count
�
_fn_kwargs
�	variables
�regularization_losses
�trainable_variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1
 
 
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
 
 
 

�0
�1
��
VARIABLE_VALUE Adam/activation_quant_14/relux/mQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_17/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_17/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_15/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_15/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_15/relux/mQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_18/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_18/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_16/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_16/relux/mQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_19/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_19/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_17/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_17/relux/mQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_20/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_20/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_18/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_18/relux/mRlayer_with_weights-12/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_noise/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_noise/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_14/relux/vQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_17/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_17/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_15/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_15/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_15/relux/vQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_18/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_18/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_16/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_16/relux/vQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_19/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_19/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_17/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_17/relux/vQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_20/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_20/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_18/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_18/relux/vRlayer_with_weights-12/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_noise/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_noise/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������@*
dtype0*$
shape:���������@
�
serving_default_input_2Placeholder*/
_output_shapes
:���������@*
dtype0*$
shape:���������@
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2activation_quant_14/reluxconv2d_noise_17/kernelconv2d_noise_17/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceactivation_quant_15/reluxconv2d_noise_18/kernelconv2d_noise_18/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceactivation_quant_16/reluxconv2d_noise_19/kernelconv2d_noise_19/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceactivation_quant_17/reluxconv2d_noise_20/kernelconv2d_noise_20/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceactivation_quant_18/reluxdense_noise/kerneldense_noise/bias*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*-
f(R&
$__inference_signature_wrapper_416307
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�#
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-activation_quant_14/relux/Read/ReadVariableOp*conv2d_noise_17/kernel/Read/ReadVariableOp(conv2d_noise_17/bias/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp-activation_quant_15/relux/Read/ReadVariableOp*conv2d_noise_18/kernel/Read/ReadVariableOp(conv2d_noise_18/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp-activation_quant_16/relux/Read/ReadVariableOp*conv2d_noise_19/kernel/Read/ReadVariableOp(conv2d_noise_19/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp-activation_quant_17/relux/Read/ReadVariableOp*conv2d_noise_20/kernel/Read/ReadVariableOp(conv2d_noise_20/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp-activation_quant_18/relux/Read/ReadVariableOp&dense_noise/kernel/Read/ReadVariableOp$dense_noise/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/activation_quant_14/relux/m/Read/ReadVariableOp1Adam/conv2d_noise_17/kernel/m/Read/ReadVariableOp/Adam/conv2d_noise_17/bias/m/Read/ReadVariableOp7Adam/batch_normalization_15/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_15/beta/m/Read/ReadVariableOp4Adam/activation_quant_15/relux/m/Read/ReadVariableOp1Adam/conv2d_noise_18/kernel/m/Read/ReadVariableOp/Adam/conv2d_noise_18/bias/m/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_16/beta/m/Read/ReadVariableOp4Adam/activation_quant_16/relux/m/Read/ReadVariableOp1Adam/conv2d_noise_19/kernel/m/Read/ReadVariableOp/Adam/conv2d_noise_19/bias/m/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_17/beta/m/Read/ReadVariableOp4Adam/activation_quant_17/relux/m/Read/ReadVariableOp1Adam/conv2d_noise_20/kernel/m/Read/ReadVariableOp/Adam/conv2d_noise_20/bias/m/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_18/beta/m/Read/ReadVariableOp4Adam/activation_quant_18/relux/m/Read/ReadVariableOp-Adam/dense_noise/kernel/m/Read/ReadVariableOp+Adam/dense_noise/bias/m/Read/ReadVariableOp4Adam/activation_quant_14/relux/v/Read/ReadVariableOp1Adam/conv2d_noise_17/kernel/v/Read/ReadVariableOp/Adam/conv2d_noise_17/bias/v/Read/ReadVariableOp7Adam/batch_normalization_15/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_15/beta/v/Read/ReadVariableOp4Adam/activation_quant_15/relux/v/Read/ReadVariableOp1Adam/conv2d_noise_18/kernel/v/Read/ReadVariableOp/Adam/conv2d_noise_18/bias/v/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_16/beta/v/Read/ReadVariableOp4Adam/activation_quant_16/relux/v/Read/ReadVariableOp1Adam/conv2d_noise_19/kernel/v/Read/ReadVariableOp/Adam/conv2d_noise_19/bias/v/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_17/beta/v/Read/ReadVariableOp4Adam/activation_quant_17/relux/v/Read/ReadVariableOp1Adam/conv2d_noise_20/kernel/v/Read/ReadVariableOp/Adam/conv2d_noise_20/bias/v/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_18/beta/v/Read/ReadVariableOp4Adam/activation_quant_18/relux/v/Read/ReadVariableOp-Adam/dense_noise/kernel/v/Read/ReadVariableOp+Adam/dense_noise/bias/v/Read/ReadVariableOpConst*a
TinZ
X2V	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: */
config_proto

CPU

GPU2*0,1J 8*(
f#R!
__inference__traced_save_418452
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameactivation_quant_14/reluxconv2d_noise_17/kernelconv2d_noise_17/biasbatch_normalization_15/gammabatch_normalization_15/beta"batch_normalization_15/moving_mean&batch_normalization_15/moving_varianceactivation_quant_15/reluxconv2d_noise_18/kernelconv2d_noise_18/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceactivation_quant_16/reluxconv2d_noise_19/kernelconv2d_noise_19/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceactivation_quant_17/reluxconv2d_noise_20/kernelconv2d_noise_20/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceactivation_quant_18/reluxdense_noise/kerneldense_noise/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount Adam/activation_quant_14/relux/mAdam/conv2d_noise_17/kernel/mAdam/conv2d_noise_17/bias/m#Adam/batch_normalization_15/gamma/m"Adam/batch_normalization_15/beta/m Adam/activation_quant_15/relux/mAdam/conv2d_noise_18/kernel/mAdam/conv2d_noise_18/bias/m#Adam/batch_normalization_16/gamma/m"Adam/batch_normalization_16/beta/m Adam/activation_quant_16/relux/mAdam/conv2d_noise_19/kernel/mAdam/conv2d_noise_19/bias/m#Adam/batch_normalization_17/gamma/m"Adam/batch_normalization_17/beta/m Adam/activation_quant_17/relux/mAdam/conv2d_noise_20/kernel/mAdam/conv2d_noise_20/bias/m#Adam/batch_normalization_18/gamma/m"Adam/batch_normalization_18/beta/m Adam/activation_quant_18/relux/mAdam/dense_noise/kernel/mAdam/dense_noise/bias/m Adam/activation_quant_14/relux/vAdam/conv2d_noise_17/kernel/vAdam/conv2d_noise_17/bias/v#Adam/batch_normalization_15/gamma/v"Adam/batch_normalization_15/beta/v Adam/activation_quant_15/relux/vAdam/conv2d_noise_18/kernel/vAdam/conv2d_noise_18/bias/v#Adam/batch_normalization_16/gamma/v"Adam/batch_normalization_16/beta/v Adam/activation_quant_16/relux/vAdam/conv2d_noise_19/kernel/vAdam/conv2d_noise_19/bias/v#Adam/batch_normalization_17/gamma/v"Adam/batch_normalization_17/beta/v Adam/activation_quant_17/relux/vAdam/conv2d_noise_20/kernel/vAdam/conv2d_noise_20/bias/v#Adam/batch_normalization_18/gamma/v"Adam/batch_normalization_18/beta/v Adam/activation_quant_18/relux/vAdam/dense_noise/kernel/vAdam/dense_noise/bias/v*`
TinY
W2U*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: */
config_proto

CPU

GPU2*0,1J 8*+
f&R$
"__inference__traced_restore_418716ر#
��
�(
__inference__traced_save_418452
file_prefix8
4savev2_activation_quant_14_relux_read_readvariableop5
1savev2_conv2d_noise_17_kernel_read_readvariableop3
/savev2_conv2d_noise_17_bias_read_readvariableop;
7savev2_batch_normalization_15_gamma_read_readvariableop:
6savev2_batch_normalization_15_beta_read_readvariableopA
=savev2_batch_normalization_15_moving_mean_read_readvariableopE
Asavev2_batch_normalization_15_moving_variance_read_readvariableop8
4savev2_activation_quant_15_relux_read_readvariableop5
1savev2_conv2d_noise_18_kernel_read_readvariableop3
/savev2_conv2d_noise_18_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop8
4savev2_activation_quant_16_relux_read_readvariableop5
1savev2_conv2d_noise_19_kernel_read_readvariableop3
/savev2_conv2d_noise_19_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop8
4savev2_activation_quant_17_relux_read_readvariableop5
1savev2_conv2d_noise_20_kernel_read_readvariableop3
/savev2_conv2d_noise_20_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop8
4savev2_activation_quant_18_relux_read_readvariableop1
-savev2_dense_noise_kernel_read_readvariableop/
+savev2_dense_noise_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_activation_quant_14_relux_m_read_readvariableop<
8savev2_adam_conv2d_noise_17_kernel_m_read_readvariableop:
6savev2_adam_conv2d_noise_17_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_15_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_15_beta_m_read_readvariableop?
;savev2_adam_activation_quant_15_relux_m_read_readvariableop<
8savev2_adam_conv2d_noise_18_kernel_m_read_readvariableop:
6savev2_adam_conv2d_noise_18_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_m_read_readvariableop?
;savev2_adam_activation_quant_16_relux_m_read_readvariableop<
8savev2_adam_conv2d_noise_19_kernel_m_read_readvariableop:
6savev2_adam_conv2d_noise_19_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_m_read_readvariableop?
;savev2_adam_activation_quant_17_relux_m_read_readvariableop<
8savev2_adam_conv2d_noise_20_kernel_m_read_readvariableop:
6savev2_adam_conv2d_noise_20_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_m_read_readvariableop?
;savev2_adam_activation_quant_18_relux_m_read_readvariableop8
4savev2_adam_dense_noise_kernel_m_read_readvariableop6
2savev2_adam_dense_noise_bias_m_read_readvariableop?
;savev2_adam_activation_quant_14_relux_v_read_readvariableop<
8savev2_adam_conv2d_noise_17_kernel_v_read_readvariableop:
6savev2_adam_conv2d_noise_17_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_15_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_15_beta_v_read_readvariableop?
;savev2_adam_activation_quant_15_relux_v_read_readvariableop<
8savev2_adam_conv2d_noise_18_kernel_v_read_readvariableop:
6savev2_adam_conv2d_noise_18_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_v_read_readvariableop?
;savev2_adam_activation_quant_16_relux_v_read_readvariableop<
8savev2_adam_conv2d_noise_19_kernel_v_read_readvariableop:
6savev2_adam_conv2d_noise_19_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_v_read_readvariableop?
;savev2_adam_activation_quant_17_relux_v_read_readvariableop<
8savev2_adam_conv2d_noise_20_kernel_v_read_readvariableop:
6savev2_adam_conv2d_noise_20_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_v_read_readvariableop?
;savev2_adam_activation_quant_18_relux_v_read_readvariableop8
4savev2_adam_dense_noise_kernel_v_read_readvariableop6
2savev2_adam_dense_noise_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_bda91e082df3415988645806ad382fc5/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�/
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�.
value�.B�.TB5layer_with_weights-0/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/relux/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/relux/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�
value�B�TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_activation_quant_14_relux_read_readvariableop1savev2_conv2d_noise_17_kernel_read_readvariableop/savev2_conv2d_noise_17_bias_read_readvariableop7savev2_batch_normalization_15_gamma_read_readvariableop6savev2_batch_normalization_15_beta_read_readvariableop=savev2_batch_normalization_15_moving_mean_read_readvariableopAsavev2_batch_normalization_15_moving_variance_read_readvariableop4savev2_activation_quant_15_relux_read_readvariableop1savev2_conv2d_noise_18_kernel_read_readvariableop/savev2_conv2d_noise_18_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop4savev2_activation_quant_16_relux_read_readvariableop1savev2_conv2d_noise_19_kernel_read_readvariableop/savev2_conv2d_noise_19_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop4savev2_activation_quant_17_relux_read_readvariableop1savev2_conv2d_noise_20_kernel_read_readvariableop/savev2_conv2d_noise_20_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop4savev2_activation_quant_18_relux_read_readvariableop-savev2_dense_noise_kernel_read_readvariableop+savev2_dense_noise_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_activation_quant_14_relux_m_read_readvariableop8savev2_adam_conv2d_noise_17_kernel_m_read_readvariableop6savev2_adam_conv2d_noise_17_bias_m_read_readvariableop>savev2_adam_batch_normalization_15_gamma_m_read_readvariableop=savev2_adam_batch_normalization_15_beta_m_read_readvariableop;savev2_adam_activation_quant_15_relux_m_read_readvariableop8savev2_adam_conv2d_noise_18_kernel_m_read_readvariableop6savev2_adam_conv2d_noise_18_bias_m_read_readvariableop>savev2_adam_batch_normalization_16_gamma_m_read_readvariableop=savev2_adam_batch_normalization_16_beta_m_read_readvariableop;savev2_adam_activation_quant_16_relux_m_read_readvariableop8savev2_adam_conv2d_noise_19_kernel_m_read_readvariableop6savev2_adam_conv2d_noise_19_bias_m_read_readvariableop>savev2_adam_batch_normalization_17_gamma_m_read_readvariableop=savev2_adam_batch_normalization_17_beta_m_read_readvariableop;savev2_adam_activation_quant_17_relux_m_read_readvariableop8savev2_adam_conv2d_noise_20_kernel_m_read_readvariableop6savev2_adam_conv2d_noise_20_bias_m_read_readvariableop>savev2_adam_batch_normalization_18_gamma_m_read_readvariableop=savev2_adam_batch_normalization_18_beta_m_read_readvariableop;savev2_adam_activation_quant_18_relux_m_read_readvariableop4savev2_adam_dense_noise_kernel_m_read_readvariableop2savev2_adam_dense_noise_bias_m_read_readvariableop;savev2_adam_activation_quant_14_relux_v_read_readvariableop8savev2_adam_conv2d_noise_17_kernel_v_read_readvariableop6savev2_adam_conv2d_noise_17_bias_v_read_readvariableop>savev2_adam_batch_normalization_15_gamma_v_read_readvariableop=savev2_adam_batch_normalization_15_beta_v_read_readvariableop;savev2_adam_activation_quant_15_relux_v_read_readvariableop8savev2_adam_conv2d_noise_18_kernel_v_read_readvariableop6savev2_adam_conv2d_noise_18_bias_v_read_readvariableop>savev2_adam_batch_normalization_16_gamma_v_read_readvariableop=savev2_adam_batch_normalization_16_beta_v_read_readvariableop;savev2_adam_activation_quant_16_relux_v_read_readvariableop8savev2_adam_conv2d_noise_19_kernel_v_read_readvariableop6savev2_adam_conv2d_noise_19_bias_v_read_readvariableop>savev2_adam_batch_normalization_17_gamma_v_read_readvariableop=savev2_adam_batch_normalization_17_beta_v_read_readvariableop;savev2_adam_activation_quant_17_relux_v_read_readvariableop8savev2_adam_conv2d_noise_20_kernel_v_read_readvariableop6savev2_adam_conv2d_noise_20_bias_v_read_readvariableop>savev2_adam_batch_normalization_18_gamma_v_read_readvariableop=savev2_adam_batch_normalization_18_beta_v_read_readvariableop;savev2_adam_activation_quant_18_relux_v_read_readvariableop4savev2_adam_dense_noise_kernel_v_read_readvariableop2savev2_adam_dense_noise_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *b
dtypesX
V2T	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : :@@:@:@:@:@:@: :@@:@:@:@:@:@: :@@:@:@:@:@:@: :@@:@:@:@:@:@: :@
:
: : : : : : : : :@@:@:@:@: :@@:@:@:@: :@@:@:@:@: :@@:@:@:@: :@
:
: :@@:@:@:@: :@@:@:@:@: :@@:@:@:@: :@@:@:@:@: :@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
R
&__inference_add_1_layer_call_fn_417501
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4153002
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�	
�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_417315
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
�
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_415018
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�	
�
G__inference_dense_noise_layer_call_and_return_conditional_losses_415747
x"
matmul_readvariableop_resource
add_readvariableop_resource
identity��MatMul/ReadVariableOp�add/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOpn
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
addX
SoftmaxSoftmaxadd:z:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_415599

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_415584
assignmovingavg_1_415591
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/415584*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/415584*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_415584*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/415584*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/415584*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_415584AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/415584*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/415591*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415591*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_415591*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415591*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415591*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_415591AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/415591*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417957

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_417942
assignmovingavg_1_417949
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/417942*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/417942*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_417942*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/417942*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/417942*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_417942AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/417942*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/417949*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417949*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_417949*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417949*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417949*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_417949AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/417949*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
N
2__inference_average_pooling2d_layer_call_fn_414882

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_4148762
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
0__inference_conv2d_noise_18_layer_call_fn_417329
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_4151962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
0__inference_conv2d_noise_17_layer_call_fn_417081
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_4150282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
f
2__inference_noise_injection_1_layer_call_fn_416976
x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallx*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_4149262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�	
�
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_417067
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417375

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_417360
assignmovingavg_1_417367
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/417360*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/417360*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_417360*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/417360*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/417360*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_417360AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/417360*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/417367*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417367*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_417367*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417367*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417367*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_417367AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/417367*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
4__inference_activation_quant_17_layer_call_fn_417783
x"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_4154972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
4__inference_activation_quant_15_layer_call_fn_417275
x"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_4151462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417979

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�	
�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_415196
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
�
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_417777
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
��
�
A__inference_model_layer_call_and_return_conditional_losses_415904
input_1
input_26
2activation_quant_14_statefulpartitionedcall_args_12
.conv2d_noise_17_statefulpartitionedcall_args_12
.conv2d_noise_17_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_19
5batch_normalization_15_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_39
5batch_normalization_15_statefulpartitionedcall_args_46
2activation_quant_15_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_46
2activation_quant_16_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_46
2activation_quant_17_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_19
5batch_normalization_18_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_39
5batch_normalization_18_statefulpartitionedcall_args_46
2activation_quant_18_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_2
identity��+activation_quant_14/StatefulPartitionedCall�;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�+activation_quant_15/StatefulPartitionedCall�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�+activation_quant_16/StatefulPartitionedCall�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�+activation_quant_17/StatefulPartitionedCall�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�+activation_quant_18/StatefulPartitionedCall�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�'conv2d_noise_17/StatefulPartitionedCall�'conv2d_noise_18/StatefulPartitionedCall�'conv2d_noise_19/StatefulPartitionedCall�'conv2d_noise_20/StatefulPartitionedCall�#dense_noise/StatefulPartitionedCall�
noise_injection/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_noise_injection_layer_call_and_return_conditional_losses_4149022!
noise_injection/PartitionedCall�
!noise_injection_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_4149302#
!noise_injection_1/PartitionedCall�
add/PartitionedCallPartitionedCall(noise_injection/PartitionedCall:output:0*noise_injection_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4149492
add/PartitionedCall�
+activation_quant_14/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:02activation_quant_14_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_14_layer_call_and_return_conditional_losses_4149782-
+activation_quant_14/StatefulPartitionedCall�
'conv2d_noise_17/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_14/StatefulPartitionedCall:output:0.conv2d_noise_17_statefulpartitionedcall_args_1.conv2d_noise_17_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_4150282)
'conv2d_noise_17/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_17/StatefulPartitionedCall:output:05batch_normalization_15_statefulpartitionedcall_args_15batch_normalization_15_statefulpartitionedcall_args_25batch_normalization_15_statefulpartitionedcall_args_35batch_normalization_15_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41510220
.batch_normalization_15/StatefulPartitionedCall�
+activation_quant_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:02activation_quant_15_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_4151462-
+activation_quant_15/StatefulPartitionedCall�
'conv2d_noise_18/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_15/StatefulPartitionedCall:output:0.conv2d_noise_18_statefulpartitionedcall_args_1.conv2d_noise_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_4151962)
'conv2d_noise_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_41527020
.batch_normalization_16/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall4activation_quant_14/StatefulPartitionedCall:output:07batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4153002
add_1/PartitionedCall�
+activation_quant_16/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:02activation_quant_16_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_4153292-
+activation_quant_16/StatefulPartitionedCall�
'conv2d_noise_19/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:0.conv2d_noise_19_statefulpartitionedcall_args_1.conv2d_noise_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_4153792)
'conv2d_noise_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41545320
.batch_normalization_17/StatefulPartitionedCall�
+activation_quant_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:02activation_quant_17_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_4154972-
+activation_quant_17/StatefulPartitionedCall�
'conv2d_noise_20/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_17/StatefulPartitionedCall:output:0.conv2d_noise_20_statefulpartitionedcall_args_1.conv2d_noise_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_4155472)
'conv2d_noise_20/StatefulPartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_20/StatefulPartitionedCall:output:05batch_normalization_18_statefulpartitionedcall_args_15batch_normalization_18_statefulpartitionedcall_args_25batch_normalization_18_statefulpartitionedcall_args_35batch_normalization_18_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_41562120
.batch_normalization_18/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:07batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4156512
add_2/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_4148762#
!average_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4156672
flatten/PartitionedCall�
+activation_quant_18/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:02activation_quant_18_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_4156952-
+activation_quant_18/StatefulPartitionedCall�
#dense_noise/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_18/StatefulPartitionedCall:output:0*dense_noise_statefulpartitionedcall_args_1*dense_noise_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_4157472%
#dense_noise/StatefulPartitionedCall�
;activation_quant_14/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_14_statefulpartitionedcall_args_1,^activation_quant_14/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_14/relux/Regularizer/SquareSquareCactivation_quant_14/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_14/relux/Regularizer/Square�
+activation_quant_14/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_14/relux/Regularizer/Const�
)activation_quant_14/relux/Regularizer/SumSum0activation_quant_14/relux/Regularizer/Square:y:04activation_quant_14/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/Sum�
+activation_quant_14/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_14/relux/Regularizer/mul/x�
)activation_quant_14/relux/Regularizer/mulMul4activation_quant_14/relux/Regularizer/mul/x:output:02activation_quant_14/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/mul�
+activation_quant_14/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_14/relux/Regularizer/add/x�
)activation_quant_14/relux/Regularizer/addAddV24activation_quant_14/relux/Regularizer/add/x:output:0-activation_quant_14/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/add�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_15_statefulpartitionedcall_args_1,^activation_quant_15/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_16_statefulpartitionedcall_args_1,^activation_quant_16/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_17_statefulpartitionedcall_args_1,^activation_quant_17/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_18_statefulpartitionedcall_args_1,^activation_quant_18/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity,dense_noise/StatefulPartitionedCall:output:0,^activation_quant_14/StatefulPartitionedCall<^activation_quant_14/relux/Regularizer/Square/ReadVariableOp,^activation_quant_15/StatefulPartitionedCall<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp,^activation_quant_16/StatefulPartitionedCall<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp,^activation_quant_17/StatefulPartitionedCall<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp,^activation_quant_18/StatefulPartitionedCall<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall(^conv2d_noise_17/StatefulPartitionedCall(^conv2d_noise_18/StatefulPartitionedCall(^conv2d_noise_19/StatefulPartitionedCall(^conv2d_noise_20/StatefulPartitionedCall$^dense_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::2Z
+activation_quant_14/StatefulPartitionedCall+activation_quant_14/StatefulPartitionedCall2z
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp;activation_quant_14/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_15/StatefulPartitionedCall+activation_quant_15/StatefulPartitionedCall2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_16/StatefulPartitionedCall+activation_quant_16/StatefulPartitionedCall2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_17/StatefulPartitionedCall+activation_quant_17/StatefulPartitionedCall2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_18/StatefulPartitionedCall+activation_quant_18/StatefulPartitionedCall2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2R
'conv2d_noise_17/StatefulPartitionedCall'conv2d_noise_17/StatefulPartitionedCall2R
'conv2d_noise_18/StatefulPartitionedCall'conv2d_noise_18/StatefulPartitionedCall2R
'conv2d_noise_19/StatefulPartitionedCall'conv2d_noise_19/StatefulPartitionedCall2R
'conv2d_noise_20/StatefulPartitionedCall'conv2d_noise_20/StatefulPartitionedCall2J
#dense_noise/StatefulPartitionedCall#dense_noise/StatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_417565
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
7__inference_batch_normalization_17_layer_call_fn_417666

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_4147002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_415379
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
�
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_418048
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpr
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yp
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*'
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0**
_input_shapes
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
7__inference_batch_normalization_16_layer_call_fn_417489

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_4145992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
0__inference_conv2d_noise_18_layer_call_fn_417322
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_4151862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_414568

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_414553
assignmovingavg_1_414560
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/414553*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/414553*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_414553*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/414553*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/414553*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_414553AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/414553*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/414560*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414560*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_414560*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414560*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414560*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_414560AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/414560*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_415369
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_415497
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417397

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417883

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_417868
assignmovingavg_1_417875
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/417868*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/417868*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_417868*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/417868*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/417868*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_417868AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/417868*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/417875*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417875*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_417875*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417875*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417875*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_417875AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/417875*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_18_layer_call_fn_417997

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4156212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
0__inference_conv2d_noise_17_layer_call_fn_417074
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_4150182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�	
�
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_415028
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�	
e
K__inference_noise_injection_layer_call_and_return_conditional_losses_414898
x
identity�?
ShapeShapex*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������@2
random_normalc
addAddV2xrandom_normal:z:0*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�
�
7__inference_batch_normalization_17_layer_call_fn_417740

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_4154312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
G__inference_dense_noise_layer_call_and_return_conditional_losses_418096
x"
matmul_readvariableop_resource
add_readvariableop_resource
identity��MatMul/ReadVariableOp�add/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOpn
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
addX
SoftmaxSoftmaxadd:z:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:! 

_user_specified_namex
�
�
7__inference_batch_normalization_15_layer_call_fn_417167

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_4144672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_18_layer_call_fn_417914

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4148322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
k
A__inference_add_1_layer_call_and_return_conditional_losses_415300

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_414863

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_414342
input_1
input_2=
9model_activation_quant_14_minimum_readvariableop_resource=
9model_conv2d_noise_17_convolution_readvariableop_resource9
5model_conv2d_noise_17_biasadd_readvariableop_resource8
4model_batch_normalization_15_readvariableop_resource:
6model_batch_normalization_15_readvariableop_1_resourceI
Emodel_batch_normalization_15_fusedbatchnormv3_readvariableop_resourceK
Gmodel_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource=
9model_activation_quant_15_minimum_readvariableop_resource=
9model_conv2d_noise_18_convolution_readvariableop_resource9
5model_conv2d_noise_18_biasadd_readvariableop_resource8
4model_batch_normalization_16_readvariableop_resource:
6model_batch_normalization_16_readvariableop_1_resourceI
Emodel_batch_normalization_16_fusedbatchnormv3_readvariableop_resourceK
Gmodel_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource=
9model_activation_quant_16_minimum_readvariableop_resource=
9model_conv2d_noise_19_convolution_readvariableop_resource9
5model_conv2d_noise_19_biasadd_readvariableop_resource8
4model_batch_normalization_17_readvariableop_resource:
6model_batch_normalization_17_readvariableop_1_resourceI
Emodel_batch_normalization_17_fusedbatchnormv3_readvariableop_resourceK
Gmodel_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource=
9model_activation_quant_17_minimum_readvariableop_resource=
9model_conv2d_noise_20_convolution_readvariableop_resource9
5model_conv2d_noise_20_biasadd_readvariableop_resource8
4model_batch_normalization_18_readvariableop_resource:
6model_batch_normalization_18_readvariableop_1_resourceI
Emodel_batch_normalization_18_fusedbatchnormv3_readvariableop_resourceK
Gmodel_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource=
9model_activation_quant_18_minimum_readvariableop_resource4
0model_dense_noise_matmul_readvariableop_resource1
-model_dense_noise_add_readvariableop_resource
identity��@model/activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp�0model/activation_quant_14/Minimum/ReadVariableOp�@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�0model/activation_quant_15/Minimum/ReadVariableOp�@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�0model/activation_quant_16/Minimum/ReadVariableOp�@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�0model/activation_quant_17/Minimum/ReadVariableOp�@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�0model/activation_quant_18/Minimum/ReadVariableOp�<model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp�>model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1�+model/batch_normalization_15/ReadVariableOp�-model/batch_normalization_15/ReadVariableOp_1�<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�+model/batch_normalization_16/ReadVariableOp�-model/batch_normalization_16/ReadVariableOp_1�<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�+model/batch_normalization_17/ReadVariableOp�-model/batch_normalization_17/ReadVariableOp_1�<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�+model/batch_normalization_18/ReadVariableOp�-model/batch_normalization_18/ReadVariableOp_1�,model/conv2d_noise_17/BiasAdd/ReadVariableOp�0model/conv2d_noise_17/convolution/ReadVariableOp�,model/conv2d_noise_18/BiasAdd/ReadVariableOp�0model/conv2d_noise_18/convolution/ReadVariableOp�,model/conv2d_noise_19/BiasAdd/ReadVariableOp�0model/conv2d_noise_19/convolution/ReadVariableOp�,model/conv2d_noise_20/BiasAdd/ReadVariableOp�0model/conv2d_noise_20/convolution/ReadVariableOp�'model/dense_noise/MatMul/ReadVariableOp�$model/dense_noise/add/ReadVariableOps
model/add/addAddV2input_1input_2*
T0*/
_output_shapes
:���������@2
model/add/add�
0model/activation_quant_14/Minimum/ReadVariableOpReadVariableOp9model_activation_quant_14_minimum_readvariableop_resource*
_output_shapes
: *
dtype022
0model/activation_quant_14/Minimum/ReadVariableOp�
!model/activation_quant_14/MinimumMinimummodel/add/add:z:08model/activation_quant_14/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_14/Minimum�
#model/activation_quant_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/activation_quant_14/Maximum/y�
!model/activation_quant_14/MaximumMaximum%model/activation_quant_14/Minimum:z:0,model/activation_quant_14/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_14/Maximum�
model/activation_quant_14/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/activation_quant_14/Const�
@model/activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9model_activation_quant_14_minimum_readvariableop_resource1^model/activation_quant_14/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02B
@model/activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp�
1model/activation_quant_14/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/activation_quant_14/Maximum:z:0(model/activation_quant_14/Const:output:0Hmodel/activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits23
1model/activation_quant_14/FakeQuantWithMinMaxVars�
0model/conv2d_noise_17/convolution/ReadVariableOpReadVariableOp9model_conv2d_noise_17_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/conv2d_noise_17/convolution/ReadVariableOp�
!model/conv2d_noise_17/convolutionConv2D;model/activation_quant_14/FakeQuantWithMinMaxVars:outputs:08model/conv2d_noise_17/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!model/conv2d_noise_17/convolution�
,model/conv2d_noise_17/BiasAdd/ReadVariableOpReadVariableOp5model_conv2d_noise_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model/conv2d_noise_17/BiasAdd/ReadVariableOp�
model/conv2d_noise_17/BiasAddBiasAdd*model/conv2d_noise_17/convolution:output:04model/conv2d_noise_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
model/conv2d_noise_17/BiasAdd�
)model/batch_normalization_15/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/batch_normalization_15/LogicalAnd/x�
)model/batch_normalization_15/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2+
)model/batch_normalization_15/LogicalAnd/y�
'model/batch_normalization_15/LogicalAnd
LogicalAnd2model/batch_normalization_15/LogicalAnd/x:output:02model/batch_normalization_15/LogicalAnd/y:output:0*
_output_shapes
: 2)
'model/batch_normalization_15/LogicalAnd�
+model/batch_normalization_15/ReadVariableOpReadVariableOp4model_batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model/batch_normalization_15/ReadVariableOp�
-model/batch_normalization_15/ReadVariableOp_1ReadVariableOp6model_batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-model/batch_normalization_15/ReadVariableOp_1�
<model/batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp�
>model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_batch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1�
-model/batch_normalization_15/FusedBatchNormV3FusedBatchNormV3&model/conv2d_noise_17/BiasAdd:output:03model/batch_normalization_15/ReadVariableOp:value:05model/batch_normalization_15/ReadVariableOp_1:value:0Dmodel/batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0Fmodel/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2/
-model/batch_normalization_15/FusedBatchNormV3�
"model/batch_normalization_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2$
"model/batch_normalization_15/Const�
0model/activation_quant_15/Minimum/ReadVariableOpReadVariableOp9model_activation_quant_15_minimum_readvariableop_resource*
_output_shapes
: *
dtype022
0model/activation_quant_15/Minimum/ReadVariableOp�
!model/activation_quant_15/MinimumMinimum1model/batch_normalization_15/FusedBatchNormV3:y:08model/activation_quant_15/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_15/Minimum�
#model/activation_quant_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/activation_quant_15/Maximum/y�
!model/activation_quant_15/MaximumMaximum%model/activation_quant_15/Minimum:z:0,model/activation_quant_15/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_15/Maximum�
model/activation_quant_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/activation_quant_15/Const�
@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9model_activation_quant_15_minimum_readvariableop_resource1^model/activation_quant_15/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02B
@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�
1model/activation_quant_15/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/activation_quant_15/Maximum:z:0(model/activation_quant_15/Const:output:0Hmodel/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits23
1model/activation_quant_15/FakeQuantWithMinMaxVars�
0model/conv2d_noise_18/convolution/ReadVariableOpReadVariableOp9model_conv2d_noise_18_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/conv2d_noise_18/convolution/ReadVariableOp�
!model/conv2d_noise_18/convolutionConv2D;model/activation_quant_15/FakeQuantWithMinMaxVars:outputs:08model/conv2d_noise_18/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!model/conv2d_noise_18/convolution�
,model/conv2d_noise_18/BiasAdd/ReadVariableOpReadVariableOp5model_conv2d_noise_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model/conv2d_noise_18/BiasAdd/ReadVariableOp�
model/conv2d_noise_18/BiasAddBiasAdd*model/conv2d_noise_18/convolution:output:04model/conv2d_noise_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
model/conv2d_noise_18/BiasAdd�
)model/batch_normalization_16/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/batch_normalization_16/LogicalAnd/x�
)model/batch_normalization_16/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2+
)model/batch_normalization_16/LogicalAnd/y�
'model/batch_normalization_16/LogicalAnd
LogicalAnd2model/batch_normalization_16/LogicalAnd/x:output:02model/batch_normalization_16/LogicalAnd/y:output:0*
_output_shapes
: 2)
'model/batch_normalization_16/LogicalAnd�
+model/batch_normalization_16/ReadVariableOpReadVariableOp4model_batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model/batch_normalization_16/ReadVariableOp�
-model/batch_normalization_16/ReadVariableOp_1ReadVariableOp6model_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-model/batch_normalization_16/ReadVariableOp_1�
<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
-model/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3&model/conv2d_noise_18/BiasAdd:output:03model/batch_normalization_16/ReadVariableOp:value:05model/batch_normalization_16/ReadVariableOp_1:value:0Dmodel/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Fmodel/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2/
-model/batch_normalization_16/FusedBatchNormV3�
"model/batch_normalization_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2$
"model/batch_normalization_16/Const�
model/add_1/addAddV2;model/activation_quant_14/FakeQuantWithMinMaxVars:outputs:01model/batch_normalization_16/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
model/add_1/add�
0model/activation_quant_16/Minimum/ReadVariableOpReadVariableOp9model_activation_quant_16_minimum_readvariableop_resource*
_output_shapes
: *
dtype022
0model/activation_quant_16/Minimum/ReadVariableOp�
!model/activation_quant_16/MinimumMinimummodel/add_1/add:z:08model/activation_quant_16/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_16/Minimum�
#model/activation_quant_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/activation_quant_16/Maximum/y�
!model/activation_quant_16/MaximumMaximum%model/activation_quant_16/Minimum:z:0,model/activation_quant_16/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_16/Maximum�
model/activation_quant_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/activation_quant_16/Const�
@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9model_activation_quant_16_minimum_readvariableop_resource1^model/activation_quant_16/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02B
@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�
1model/activation_quant_16/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/activation_quant_16/Maximum:z:0(model/activation_quant_16/Const:output:0Hmodel/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits23
1model/activation_quant_16/FakeQuantWithMinMaxVars�
0model/conv2d_noise_19/convolution/ReadVariableOpReadVariableOp9model_conv2d_noise_19_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/conv2d_noise_19/convolution/ReadVariableOp�
!model/conv2d_noise_19/convolutionConv2D;model/activation_quant_16/FakeQuantWithMinMaxVars:outputs:08model/conv2d_noise_19/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!model/conv2d_noise_19/convolution�
,model/conv2d_noise_19/BiasAdd/ReadVariableOpReadVariableOp5model_conv2d_noise_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model/conv2d_noise_19/BiasAdd/ReadVariableOp�
model/conv2d_noise_19/BiasAddBiasAdd*model/conv2d_noise_19/convolution:output:04model/conv2d_noise_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
model/conv2d_noise_19/BiasAdd�
)model/batch_normalization_17/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/batch_normalization_17/LogicalAnd/x�
)model/batch_normalization_17/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2+
)model/batch_normalization_17/LogicalAnd/y�
'model/batch_normalization_17/LogicalAnd
LogicalAnd2model/batch_normalization_17/LogicalAnd/x:output:02model/batch_normalization_17/LogicalAnd/y:output:0*
_output_shapes
: 2)
'model/batch_normalization_17/LogicalAnd�
+model/batch_normalization_17/ReadVariableOpReadVariableOp4model_batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model/batch_normalization_17/ReadVariableOp�
-model/batch_normalization_17/ReadVariableOp_1ReadVariableOp6model_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-model/batch_normalization_17/ReadVariableOp_1�
<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
-model/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3&model/conv2d_noise_19/BiasAdd:output:03model/batch_normalization_17/ReadVariableOp:value:05model/batch_normalization_17/ReadVariableOp_1:value:0Dmodel/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Fmodel/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2/
-model/batch_normalization_17/FusedBatchNormV3�
"model/batch_normalization_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2$
"model/batch_normalization_17/Const�
0model/activation_quant_17/Minimum/ReadVariableOpReadVariableOp9model_activation_quant_17_minimum_readvariableop_resource*
_output_shapes
: *
dtype022
0model/activation_quant_17/Minimum/ReadVariableOp�
!model/activation_quant_17/MinimumMinimum1model/batch_normalization_17/FusedBatchNormV3:y:08model/activation_quant_17/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_17/Minimum�
#model/activation_quant_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/activation_quant_17/Maximum/y�
!model/activation_quant_17/MaximumMaximum%model/activation_quant_17/Minimum:z:0,model/activation_quant_17/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_17/Maximum�
model/activation_quant_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/activation_quant_17/Const�
@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9model_activation_quant_17_minimum_readvariableop_resource1^model/activation_quant_17/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02B
@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�
1model/activation_quant_17/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/activation_quant_17/Maximum:z:0(model/activation_quant_17/Const:output:0Hmodel/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits23
1model/activation_quant_17/FakeQuantWithMinMaxVars�
0model/conv2d_noise_20/convolution/ReadVariableOpReadVariableOp9model_conv2d_noise_20_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/conv2d_noise_20/convolution/ReadVariableOp�
!model/conv2d_noise_20/convolutionConv2D;model/activation_quant_17/FakeQuantWithMinMaxVars:outputs:08model/conv2d_noise_20/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!model/conv2d_noise_20/convolution�
,model/conv2d_noise_20/BiasAdd/ReadVariableOpReadVariableOp5model_conv2d_noise_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model/conv2d_noise_20/BiasAdd/ReadVariableOp�
model/conv2d_noise_20/BiasAddBiasAdd*model/conv2d_noise_20/convolution:output:04model/conv2d_noise_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
model/conv2d_noise_20/BiasAdd�
)model/batch_normalization_18/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/batch_normalization_18/LogicalAnd/x�
)model/batch_normalization_18/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2+
)model/batch_normalization_18/LogicalAnd/y�
'model/batch_normalization_18/LogicalAnd
LogicalAnd2model/batch_normalization_18/LogicalAnd/x:output:02model/batch_normalization_18/LogicalAnd/y:output:0*
_output_shapes
: 2)
'model/batch_normalization_18/LogicalAnd�
+model/batch_normalization_18/ReadVariableOpReadVariableOp4model_batch_normalization_18_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model/batch_normalization_18/ReadVariableOp�
-model/batch_normalization_18/ReadVariableOp_1ReadVariableOp6model_batch_normalization_18_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-model/batch_normalization_18/ReadVariableOp_1�
<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
-model/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3&model/conv2d_noise_20/BiasAdd:output:03model/batch_normalization_18/ReadVariableOp:value:05model/batch_normalization_18/ReadVariableOp_1:value:0Dmodel/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Fmodel/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2/
-model/batch_normalization_18/FusedBatchNormV3�
"model/batch_normalization_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2$
"model/batch_normalization_18/Const�
model/add_2/addAddV2;model/activation_quant_16/FakeQuantWithMinMaxVars:outputs:01model/batch_normalization_18/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
model/add_2/add�
model/average_pooling2d/AvgPoolAvgPoolmodel/add_2/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2!
model/average_pooling2d/AvgPool{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
model/flatten/Const�
model/flatten/ReshapeReshape(model/average_pooling2d/AvgPool:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:���������@2
model/flatten/Reshape�
0model/activation_quant_18/Minimum/ReadVariableOpReadVariableOp9model_activation_quant_18_minimum_readvariableop_resource*
_output_shapes
: *
dtype022
0model/activation_quant_18/Minimum/ReadVariableOp�
!model/activation_quant_18/MinimumMinimummodel/flatten/Reshape:output:08model/activation_quant_18/Minimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2#
!model/activation_quant_18/Minimum�
#model/activation_quant_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/activation_quant_18/Maximum/y�
!model/activation_quant_18/MaximumMaximum%model/activation_quant_18/Minimum:z:0,model/activation_quant_18/Maximum/y:output:0*
T0*'
_output_shapes
:���������@2#
!model/activation_quant_18/Maximum�
model/activation_quant_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/activation_quant_18/Const�
@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9model_activation_quant_18_minimum_readvariableop_resource1^model/activation_quant_18/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02B
@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�
1model/activation_quant_18/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/activation_quant_18/Maximum:z:0(model/activation_quant_18/Const:output:0Hmodel/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits23
1model/activation_quant_18/FakeQuantWithMinMaxVars�
'model/dense_noise/MatMul/ReadVariableOpReadVariableOp0model_dense_noise_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02)
'model/dense_noise/MatMul/ReadVariableOp�
model/dense_noise/MatMulMatMul;model/activation_quant_18/FakeQuantWithMinMaxVars:outputs:0/model/dense_noise/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
model/dense_noise/MatMul�
$model/dense_noise/add/ReadVariableOpReadVariableOp-model_dense_noise_add_readvariableop_resource*
_output_shapes
:
*
dtype02&
$model/dense_noise/add/ReadVariableOp�
model/dense_noise/addAddV2"model/dense_noise/MatMul:product:0,model/dense_noise/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
model/dense_noise/add�
model/dense_noise/SoftmaxSoftmaxmodel/dense_noise/add:z:0*
T0*'
_output_shapes
:���������
2
model/dense_noise/Softmax�
IdentityIdentity#model/dense_noise/Softmax:softmax:0A^model/activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp1^model/activation_quant_14/Minimum/ReadVariableOpA^model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp1^model/activation_quant_15/Minimum/ReadVariableOpA^model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp1^model/activation_quant_16/Minimum/ReadVariableOpA^model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp1^model/activation_quant_17/Minimum/ReadVariableOpA^model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp1^model/activation_quant_18/Minimum/ReadVariableOp=^model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp?^model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1,^model/batch_normalization_15/ReadVariableOp.^model/batch_normalization_15/ReadVariableOp_1=^model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?^model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1,^model/batch_normalization_16/ReadVariableOp.^model/batch_normalization_16/ReadVariableOp_1=^model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?^model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1,^model/batch_normalization_17/ReadVariableOp.^model/batch_normalization_17/ReadVariableOp_1=^model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp?^model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1,^model/batch_normalization_18/ReadVariableOp.^model/batch_normalization_18/ReadVariableOp_1-^model/conv2d_noise_17/BiasAdd/ReadVariableOp1^model/conv2d_noise_17/convolution/ReadVariableOp-^model/conv2d_noise_18/BiasAdd/ReadVariableOp1^model/conv2d_noise_18/convolution/ReadVariableOp-^model/conv2d_noise_19/BiasAdd/ReadVariableOp1^model/conv2d_noise_19/convolution/ReadVariableOp-^model/conv2d_noise_20/BiasAdd/ReadVariableOp1^model/conv2d_noise_20/convolution/ReadVariableOp(^model/dense_noise/MatMul/ReadVariableOp%^model/dense_noise/add/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::2�
@model/activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp@model/activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp2d
0model/activation_quant_14/Minimum/ReadVariableOp0model/activation_quant_14/Minimum/ReadVariableOp2�
@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp2d
0model/activation_quant_15/Minimum/ReadVariableOp0model/activation_quant_15/Minimum/ReadVariableOp2�
@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp2d
0model/activation_quant_16/Minimum/ReadVariableOp0model/activation_quant_16/Minimum/ReadVariableOp2�
@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp2d
0model/activation_quant_17/Minimum/ReadVariableOp0model/activation_quant_17/Minimum/ReadVariableOp2�
@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp2d
0model/activation_quant_18/Minimum/ReadVariableOp0model/activation_quant_18/Minimum/ReadVariableOp2|
<model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp<model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp2�
>model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1>model/batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12Z
+model/batch_normalization_15/ReadVariableOp+model/batch_normalization_15/ReadVariableOp2^
-model/batch_normalization_15/ReadVariableOp_1-model/batch_normalization_15/ReadVariableOp_12|
<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12Z
+model/batch_normalization_16/ReadVariableOp+model/batch_normalization_16/ReadVariableOp2^
-model/batch_normalization_16/ReadVariableOp_1-model/batch_normalization_16/ReadVariableOp_12|
<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12Z
+model/batch_normalization_17/ReadVariableOp+model/batch_normalization_17/ReadVariableOp2^
-model/batch_normalization_17/ReadVariableOp_1-model/batch_normalization_17/ReadVariableOp_12|
<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12Z
+model/batch_normalization_18/ReadVariableOp+model/batch_normalization_18/ReadVariableOp2^
-model/batch_normalization_18/ReadVariableOp_1-model/batch_normalization_18/ReadVariableOp_12\
,model/conv2d_noise_17/BiasAdd/ReadVariableOp,model/conv2d_noise_17/BiasAdd/ReadVariableOp2d
0model/conv2d_noise_17/convolution/ReadVariableOp0model/conv2d_noise_17/convolution/ReadVariableOp2\
,model/conv2d_noise_18/BiasAdd/ReadVariableOp,model/conv2d_noise_18/BiasAdd/ReadVariableOp2d
0model/conv2d_noise_18/convolution/ReadVariableOp0model/conv2d_noise_18/convolution/ReadVariableOp2\
,model/conv2d_noise_19/BiasAdd/ReadVariableOp,model/conv2d_noise_19/BiasAdd/ReadVariableOp2d
0model/conv2d_noise_19/convolution/ReadVariableOp0model/conv2d_noise_19/convolution/ReadVariableOp2\
,model/conv2d_noise_20/BiasAdd/ReadVariableOp,model/conv2d_noise_20/BiasAdd/ReadVariableOp2d
0model/conv2d_noise_20/convolution/ReadVariableOp0model/conv2d_noise_20/convolution/ReadVariableOp2R
'model/dense_noise/MatMul/ReadVariableOp'model/dense_noise/MatMul/ReadVariableOp2L
$model/dense_noise/add/ReadVariableOp$model/dense_noise/add/ReadVariableOp:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
0__inference_conv2d_noise_20_layer_call_fn_417830
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_4155372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�	
g
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_414926
x
identity�?
ShapeShapex*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������@2
random_normalc
addAddV2xrandom_normal:z:0*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417127

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_417112
assignmovingavg_1_417119
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/417112*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/417112*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_417112*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/417112*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/417112*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_417112AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/417112*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/417119*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417119*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_417119*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417119*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417119*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_417119AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/417119*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417709

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_417694
assignmovingavg_1_417701
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/417694*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/417694*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_417694*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/417694*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/417694*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_417694AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/417694*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/417701*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417701*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_417701*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417701*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417701*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_417701AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/417701*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_415453

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�	
&__inference_model_layer_call_fn_416894
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4160052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
G
0__inference_noise_injection_layer_call_fn_416956
x
identity�
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_noise_injection_layer_call_and_return_conditional_losses_4149022
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�
�
,__inference_dense_noise_layer_call_fn_418110
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_4157472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
G__inference_dense_noise_layer_call_and_return_conditional_losses_418085
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*
_output_shapes

:@
*
dtype02
Abs/ReadVariableOpV
AbsAbsAbs/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
Abs_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   
   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@
*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*
_output_shapes

:@
2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@
2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*
_output_shapes

:@
*
dtype02
ReadVariableOpg
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*
_output_shapes

:@
2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:
*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:
2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:
2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:
2
add_1X
MatMulMatMulxadd:z:0*
T0*'
_output_shapes
:���������
2
MatMulf
add_2AddV2MatMul:product:0	add_1:z:0*
T0*'
_output_shapes
:���������
2
add_2Z
SoftmaxSoftmax	add_2:z:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
4__inference_activation_quant_18_layer_call_fn_418054
x"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_4156952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0**
_input_shapes
:���������@:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
7__inference_batch_normalization_18_layer_call_fn_417988

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4155992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
��
�
A__inference_model_layer_call_and_return_conditional_losses_416139

inputs
inputs_16
2activation_quant_14_statefulpartitionedcall_args_12
.conv2d_noise_17_statefulpartitionedcall_args_12
.conv2d_noise_17_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_19
5batch_normalization_15_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_39
5batch_normalization_15_statefulpartitionedcall_args_46
2activation_quant_15_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_46
2activation_quant_16_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_46
2activation_quant_17_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_19
5batch_normalization_18_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_39
5batch_normalization_18_statefulpartitionedcall_args_46
2activation_quant_18_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_2
identity��+activation_quant_14/StatefulPartitionedCall�;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�+activation_quant_15/StatefulPartitionedCall�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�+activation_quant_16/StatefulPartitionedCall�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�+activation_quant_17/StatefulPartitionedCall�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�+activation_quant_18/StatefulPartitionedCall�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�'conv2d_noise_17/StatefulPartitionedCall�'conv2d_noise_18/StatefulPartitionedCall�'conv2d_noise_19/StatefulPartitionedCall�'conv2d_noise_20/StatefulPartitionedCall�#dense_noise/StatefulPartitionedCall�
noise_injection/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_noise_injection_layer_call_and_return_conditional_losses_4149022!
noise_injection/PartitionedCall�
!noise_injection_1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_4149302#
!noise_injection_1/PartitionedCall�
add/PartitionedCallPartitionedCall(noise_injection/PartitionedCall:output:0*noise_injection_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4149492
add/PartitionedCall�
+activation_quant_14/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:02activation_quant_14_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_14_layer_call_and_return_conditional_losses_4149782-
+activation_quant_14/StatefulPartitionedCall�
'conv2d_noise_17/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_14/StatefulPartitionedCall:output:0.conv2d_noise_17_statefulpartitionedcall_args_1.conv2d_noise_17_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_4150282)
'conv2d_noise_17/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_17/StatefulPartitionedCall:output:05batch_normalization_15_statefulpartitionedcall_args_15batch_normalization_15_statefulpartitionedcall_args_25batch_normalization_15_statefulpartitionedcall_args_35batch_normalization_15_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41510220
.batch_normalization_15/StatefulPartitionedCall�
+activation_quant_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:02activation_quant_15_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_4151462-
+activation_quant_15/StatefulPartitionedCall�
'conv2d_noise_18/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_15/StatefulPartitionedCall:output:0.conv2d_noise_18_statefulpartitionedcall_args_1.conv2d_noise_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_4151962)
'conv2d_noise_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_41527020
.batch_normalization_16/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall4activation_quant_14/StatefulPartitionedCall:output:07batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4153002
add_1/PartitionedCall�
+activation_quant_16/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:02activation_quant_16_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_4153292-
+activation_quant_16/StatefulPartitionedCall�
'conv2d_noise_19/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:0.conv2d_noise_19_statefulpartitionedcall_args_1.conv2d_noise_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_4153792)
'conv2d_noise_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41545320
.batch_normalization_17/StatefulPartitionedCall�
+activation_quant_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:02activation_quant_17_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_4154972-
+activation_quant_17/StatefulPartitionedCall�
'conv2d_noise_20/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_17/StatefulPartitionedCall:output:0.conv2d_noise_20_statefulpartitionedcall_args_1.conv2d_noise_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_4155472)
'conv2d_noise_20/StatefulPartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_20/StatefulPartitionedCall:output:05batch_normalization_18_statefulpartitionedcall_args_15batch_normalization_18_statefulpartitionedcall_args_25batch_normalization_18_statefulpartitionedcall_args_35batch_normalization_18_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_41562120
.batch_normalization_18/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:07batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4156512
add_2/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_4148762#
!average_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4156672
flatten/PartitionedCall�
+activation_quant_18/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:02activation_quant_18_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_4156952-
+activation_quant_18/StatefulPartitionedCall�
#dense_noise/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_18/StatefulPartitionedCall:output:0*dense_noise_statefulpartitionedcall_args_1*dense_noise_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_4157472%
#dense_noise/StatefulPartitionedCall�
;activation_quant_14/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_14_statefulpartitionedcall_args_1,^activation_quant_14/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_14/relux/Regularizer/SquareSquareCactivation_quant_14/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_14/relux/Regularizer/Square�
+activation_quant_14/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_14/relux/Regularizer/Const�
)activation_quant_14/relux/Regularizer/SumSum0activation_quant_14/relux/Regularizer/Square:y:04activation_quant_14/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/Sum�
+activation_quant_14/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_14/relux/Regularizer/mul/x�
)activation_quant_14/relux/Regularizer/mulMul4activation_quant_14/relux/Regularizer/mul/x:output:02activation_quant_14/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/mul�
+activation_quant_14/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_14/relux/Regularizer/add/x�
)activation_quant_14/relux/Regularizer/addAddV24activation_quant_14/relux/Regularizer/add/x:output:0-activation_quant_14/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/add�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_15_statefulpartitionedcall_args_1,^activation_quant_15/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_16_statefulpartitionedcall_args_1,^activation_quant_16/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_17_statefulpartitionedcall_args_1,^activation_quant_17/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_18_statefulpartitionedcall_args_1,^activation_quant_18/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity,dense_noise/StatefulPartitionedCall:output:0,^activation_quant_14/StatefulPartitionedCall<^activation_quant_14/relux/Regularizer/Square/ReadVariableOp,^activation_quant_15/StatefulPartitionedCall<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp,^activation_quant_16/StatefulPartitionedCall<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp,^activation_quant_17/StatefulPartitionedCall<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp,^activation_quant_18/StatefulPartitionedCall<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall(^conv2d_noise_17/StatefulPartitionedCall(^conv2d_noise_18/StatefulPartitionedCall(^conv2d_noise_19/StatefulPartitionedCall(^conv2d_noise_20/StatefulPartitionedCall$^dense_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::2Z
+activation_quant_14/StatefulPartitionedCall+activation_quant_14/StatefulPartitionedCall2z
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp;activation_quant_14/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_15/StatefulPartitionedCall+activation_quant_15/StatefulPartitionedCall2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_16/StatefulPartitionedCall+activation_quant_16/StatefulPartitionedCall2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_17/StatefulPartitionedCall+activation_quant_17/StatefulPartitionedCall2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_18/StatefulPartitionedCall+activation_quant_18/StatefulPartitionedCall2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2R
'conv2d_noise_17/StatefulPartitionedCall'conv2d_noise_17/StatefulPartitionedCall2R
'conv2d_noise_18/StatefulPartitionedCall'conv2d_noise_18/StatefulPartitionedCall2R
'conv2d_noise_19/StatefulPartitionedCall'conv2d_noise_19/StatefulPartitionedCall2R
'conv2d_noise_20/StatefulPartitionedCall'conv2d_noise_20/StatefulPartitionedCall2J
#dense_noise/StatefulPartitionedCall#dense_noise/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_418136H
Dactivation_quant_15_relux_regularizer_square_readvariableop_resource
identity��;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOpDactivation_quant_15_relux_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
IdentityIdentity-activation_quant_15/relux/Regularizer/add:z:0<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp
�
�
7__inference_batch_normalization_15_layer_call_fn_417232

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_4150802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417471

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
d
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_414930
x
identity]
IdentityIdentityx*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417223

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
d
0__inference_noise_injection_layer_call_fn_416951
x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallx*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_noise_injection_layer_call_and_return_conditional_losses_4148982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_414467

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_17_layer_call_fn_417675

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_4147312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_418015

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_16_layer_call_fn_417480

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_4145682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_16_layer_call_fn_417415

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_4152702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�	
&__inference_model_layer_call_fn_416039
input_1
input_2"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinput_1input_2statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4160052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
__inference_loss_fn_0_418123H
Dactivation_quant_14_relux_regularizer_square_readvariableop_resource
identity��;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
;activation_quant_14/relux/Regularizer/Square/ReadVariableOpReadVariableOpDactivation_quant_14_relux_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_14/relux/Regularizer/SquareSquareCactivation_quant_14/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_14/relux/Regularizer/Square�
+activation_quant_14/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_14/relux/Regularizer/Const�
)activation_quant_14/relux/Regularizer/SumSum0activation_quant_14/relux/Regularizer/Square:y:04activation_quant_14/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/Sum�
+activation_quant_14/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_14/relux/Regularizer/mul/x�
)activation_quant_14/relux/Regularizer/mulMul4activation_quant_14/relux/Regularizer/mul/x:output:02activation_quant_14/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/mul�
+activation_quant_14/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_14/relux/Regularizer/add/x�
)activation_quant_14/relux/Regularizer/addAddV24activation_quant_14/relux/Regularizer/add/x:output:0-activation_quant_14/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/add�
IdentityIdentity-activation_quant_14/relux/Regularizer/add:z:0<^activation_quant_14/relux/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp;activation_quant_14/relux/Regularizer/Square/ReadVariableOp
�$
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_415248

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_415233
assignmovingavg_1_415240
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/415233*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/415233*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_415233*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/415233*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/415233*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_415233AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/415233*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/415240*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415240*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_415240*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415240*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415240*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_415240AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/415240*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_417813
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
__inference_loss_fn_4_418175H
Dactivation_quant_18_relux_regularizer_square_readvariableop_resource
identity��;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOpDactivation_quant_18_relux_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity-activation_quant_18/relux/Regularizer/add:z:0<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_415667

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_415695
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpr
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yp
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*'
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0**
_input_shapes
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
0__inference_conv2d_noise_19_layer_call_fn_417582
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_4153692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
R
&__inference_add_2_layer_call_fn_418009
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4156512
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_415102

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_414731

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�	
�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_417575
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
�
,__inference_dense_noise_layer_call_fn_418103
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_4157362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417149

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
��
�
A__inference_model_layer_call_and_return_conditional_losses_416857
inputs_0
inputs_17
3activation_quant_14_minimum_readvariableop_resource7
3conv2d_noise_17_convolution_readvariableop_resource3
/conv2d_noise_17_biasadd_readvariableop_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resourceC
?batch_normalization_15_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource7
3activation_quant_15_minimum_readvariableop_resource7
3conv2d_noise_18_convolution_readvariableop_resource3
/conv2d_noise_18_biasadd_readvariableop_resource2
.batch_normalization_16_readvariableop_resource4
0batch_normalization_16_readvariableop_1_resourceC
?batch_normalization_16_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource7
3activation_quant_16_minimum_readvariableop_resource7
3conv2d_noise_19_convolution_readvariableop_resource3
/conv2d_noise_19_biasadd_readvariableop_resource2
.batch_normalization_17_readvariableop_resource4
0batch_normalization_17_readvariableop_1_resourceC
?batch_normalization_17_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource7
3activation_quant_17_minimum_readvariableop_resource7
3conv2d_noise_20_convolution_readvariableop_resource3
/conv2d_noise_20_biasadd_readvariableop_resource2
.batch_normalization_18_readvariableop_resource4
0batch_normalization_18_readvariableop_1_resourceC
?batch_normalization_18_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource7
3activation_quant_18_minimum_readvariableop_resource.
*dense_noise_matmul_readvariableop_resource+
'dense_noise_add_readvariableop_resource
identity��:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_14/Minimum/ReadVariableOp�;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_15/Minimum/ReadVariableOp�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_16/Minimum/ReadVariableOp�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_17/Minimum/ReadVariableOp�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_18/Minimum/ReadVariableOp�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�6batch_normalization_15/FusedBatchNormV3/ReadVariableOp�8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_15/ReadVariableOp�'batch_normalization_15/ReadVariableOp_1�6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_16/ReadVariableOp�'batch_normalization_16/ReadVariableOp_1�6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_17/ReadVariableOp�'batch_normalization_17/ReadVariableOp_1�6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_18/ReadVariableOp�'batch_normalization_18/ReadVariableOp_1�&conv2d_noise_17/BiasAdd/ReadVariableOp�*conv2d_noise_17/convolution/ReadVariableOp�&conv2d_noise_18/BiasAdd/ReadVariableOp�*conv2d_noise_18/convolution/ReadVariableOp�&conv2d_noise_19/BiasAdd/ReadVariableOp�*conv2d_noise_19/convolution/ReadVariableOp�&conv2d_noise_20/BiasAdd/ReadVariableOp�*conv2d_noise_20/convolution/ReadVariableOp�!dense_noise/MatMul/ReadVariableOp�dense_noise/add/ReadVariableOpi
add/addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������@2	
add/add�
*activation_quant_14/Minimum/ReadVariableOpReadVariableOp3activation_quant_14_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_14/Minimum/ReadVariableOp�
activation_quant_14/MinimumMinimumadd/add:z:02activation_quant_14/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_14/Minimum�
activation_quant_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_14/Maximum/y�
activation_quant_14/MaximumMaximumactivation_quant_14/Minimum:z:0&activation_quant_14/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_14/Maximum{
activation_quant_14/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_14/Const�
:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_14_minimum_readvariableop_resource+^activation_quant_14/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_14/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_14/Maximum:z:0"activation_quant_14/Const:output:0Bactivation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_14/FakeQuantWithMinMaxVars�
*conv2d_noise_17/convolution/ReadVariableOpReadVariableOp3conv2d_noise_17_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*conv2d_noise_17/convolution/ReadVariableOp�
conv2d_noise_17/convolutionConv2D5activation_quant_14/FakeQuantWithMinMaxVars:outputs:02conv2d_noise_17/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_17/convolution�
&conv2d_noise_17/BiasAdd/ReadVariableOpReadVariableOp/conv2d_noise_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&conv2d_noise_17/BiasAdd/ReadVariableOp�
conv2d_noise_17/BiasAddBiasAdd$conv2d_noise_17/convolution:output:0.conv2d_noise_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_17/BiasAdd�
#batch_normalization_15/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_15/LogicalAnd/x�
#batch_normalization_15/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_15/LogicalAnd/y�
!batch_normalization_15/LogicalAnd
LogicalAnd,batch_normalization_15/LogicalAnd/x:output:0,batch_normalization_15/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_15/LogicalAnd�
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_15/ReadVariableOp�
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_15/ReadVariableOp_1�
6batch_normalization_15/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_15_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_15_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_17/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0>batch_normalization_15/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_15/FusedBatchNormV3�
batch_normalization_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_15/Const�
*activation_quant_15/Minimum/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_15/Minimum/ReadVariableOp�
activation_quant_15/MinimumMinimum+batch_normalization_15/FusedBatchNormV3:y:02activation_quant_15/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_15/Minimum�
activation_quant_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_15/Maximum/y�
activation_quant_15/MaximumMaximumactivation_quant_15/Minimum:z:0&activation_quant_15/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_15/Maximum{
activation_quant_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_15/Const�
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource+^activation_quant_15/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_15/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_15/Maximum:z:0"activation_quant_15/Const:output:0Bactivation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_15/FakeQuantWithMinMaxVars�
*conv2d_noise_18/convolution/ReadVariableOpReadVariableOp3conv2d_noise_18_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*conv2d_noise_18/convolution/ReadVariableOp�
conv2d_noise_18/convolutionConv2D5activation_quant_15/FakeQuantWithMinMaxVars:outputs:02conv2d_noise_18/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_18/convolution�
&conv2d_noise_18/BiasAdd/ReadVariableOpReadVariableOp/conv2d_noise_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&conv2d_noise_18/BiasAdd/ReadVariableOp�
conv2d_noise_18/BiasAddBiasAdd$conv2d_noise_18/convolution:output:0.conv2d_noise_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_18/BiasAdd�
#batch_normalization_16/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_16/LogicalAnd/x�
#batch_normalization_16/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_16/LogicalAnd/y�
!batch_normalization_16/LogicalAnd
LogicalAnd,batch_normalization_16/LogicalAnd/x:output:0,batch_normalization_16/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_16/LogicalAnd�
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOp�
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1�
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_18/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3�
batch_normalization_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_16/Const�
	add_1/addAddV25activation_quant_14/FakeQuantWithMinMaxVars:outputs:0+batch_normalization_16/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
	add_1/add�
*activation_quant_16/Minimum/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_16/Minimum/ReadVariableOp�
activation_quant_16/MinimumMinimumadd_1/add:z:02activation_quant_16/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_16/Minimum�
activation_quant_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_16/Maximum/y�
activation_quant_16/MaximumMaximumactivation_quant_16/Minimum:z:0&activation_quant_16/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_16/Maximum{
activation_quant_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_16/Const�
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource+^activation_quant_16/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_16/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_16/Maximum:z:0"activation_quant_16/Const:output:0Bactivation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_16/FakeQuantWithMinMaxVars�
*conv2d_noise_19/convolution/ReadVariableOpReadVariableOp3conv2d_noise_19_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*conv2d_noise_19/convolution/ReadVariableOp�
conv2d_noise_19/convolutionConv2D5activation_quant_16/FakeQuantWithMinMaxVars:outputs:02conv2d_noise_19/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_19/convolution�
&conv2d_noise_19/BiasAdd/ReadVariableOpReadVariableOp/conv2d_noise_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&conv2d_noise_19/BiasAdd/ReadVariableOp�
conv2d_noise_19/BiasAddBiasAdd$conv2d_noise_19/convolution:output:0.conv2d_noise_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_19/BiasAdd�
#batch_normalization_17/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_17/LogicalAnd/x�
#batch_normalization_17/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_17/LogicalAnd/y�
!batch_normalization_17/LogicalAnd
LogicalAnd,batch_normalization_17/LogicalAnd/x:output:0,batch_normalization_17/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_17/LogicalAnd�
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_17/ReadVariableOp�
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_17/ReadVariableOp_1�
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_19/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3�
batch_normalization_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_17/Const�
*activation_quant_17/Minimum/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_17/Minimum/ReadVariableOp�
activation_quant_17/MinimumMinimum+batch_normalization_17/FusedBatchNormV3:y:02activation_quant_17/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_17/Minimum�
activation_quant_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_17/Maximum/y�
activation_quant_17/MaximumMaximumactivation_quant_17/Minimum:z:0&activation_quant_17/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_17/Maximum{
activation_quant_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_17/Const�
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource+^activation_quant_17/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_17/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_17/Maximum:z:0"activation_quant_17/Const:output:0Bactivation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_17/FakeQuantWithMinMaxVars�
*conv2d_noise_20/convolution/ReadVariableOpReadVariableOp3conv2d_noise_20_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*conv2d_noise_20/convolution/ReadVariableOp�
conv2d_noise_20/convolutionConv2D5activation_quant_17/FakeQuantWithMinMaxVars:outputs:02conv2d_noise_20/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_20/convolution�
&conv2d_noise_20/BiasAdd/ReadVariableOpReadVariableOp/conv2d_noise_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&conv2d_noise_20/BiasAdd/ReadVariableOp�
conv2d_noise_20/BiasAddBiasAdd$conv2d_noise_20/convolution:output:0.conv2d_noise_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_20/BiasAdd�
#batch_normalization_18/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_18/LogicalAnd/x�
#batch_normalization_18/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_18/LogicalAnd/y�
!batch_normalization_18/LogicalAnd
LogicalAnd,batch_normalization_18/LogicalAnd/x:output:0,batch_normalization_18/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_18/LogicalAnd�
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_18/ReadVariableOp�
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_18/ReadVariableOp_1�
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_20/BiasAdd:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_18/FusedBatchNormV3�
batch_normalization_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_18/Const�
	add_2/addAddV25activation_quant_16/FakeQuantWithMinMaxVars:outputs:0+batch_normalization_18/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
	add_2/add�
average_pooling2d/AvgPoolAvgPooladd_2/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
flatten/Const�
flatten/ReshapeReshape"average_pooling2d/AvgPool:output:0flatten/Const:output:0*
T0*'
_output_shapes
:���������@2
flatten/Reshape�
*activation_quant_18/Minimum/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_18/Minimum/ReadVariableOp�
activation_quant_18/MinimumMinimumflatten/Reshape:output:02activation_quant_18/Minimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
activation_quant_18/Minimum�
activation_quant_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_18/Maximum/y�
activation_quant_18/MaximumMaximumactivation_quant_18/Minimum:z:0&activation_quant_18/Maximum/y:output:0*
T0*'
_output_shapes
:���������@2
activation_quant_18/Maximum{
activation_quant_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_18/Const�
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource+^activation_quant_18/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_18/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_18/Maximum:z:0"activation_quant_18/Const:output:0Bactivation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits2-
+activation_quant_18/FakeQuantWithMinMaxVars�
!dense_noise/MatMul/ReadVariableOpReadVariableOp*dense_noise_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02#
!dense_noise/MatMul/ReadVariableOp�
dense_noise/MatMulMatMul5activation_quant_18/FakeQuantWithMinMaxVars:outputs:0)dense_noise/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_noise/MatMul�
dense_noise/add/ReadVariableOpReadVariableOp'dense_noise_add_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_noise/add/ReadVariableOp�
dense_noise/addAddV2dense_noise/MatMul:product:0&dense_noise/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_noise/add|
dense_noise/SoftmaxSoftmaxdense_noise/add:z:0*
T0*'
_output_shapes
:���������
2
dense_noise/Softmax�
;activation_quant_14/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_14_minimum_readvariableop_resource;^activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_14/relux/Regularizer/SquareSquareCactivation_quant_14/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_14/relux/Regularizer/Square�
+activation_quant_14/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_14/relux/Regularizer/Const�
)activation_quant_14/relux/Regularizer/SumSum0activation_quant_14/relux/Regularizer/Square:y:04activation_quant_14/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/Sum�
+activation_quant_14/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_14/relux/Regularizer/mul/x�
)activation_quant_14/relux/Regularizer/mulMul4activation_quant_14/relux/Regularizer/mul/x:output:02activation_quant_14/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/mul�
+activation_quant_14/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_14/relux/Regularizer/add/x�
)activation_quant_14/relux/Regularizer/addAddV24activation_quant_14/relux/Regularizer/add/x:output:0-activation_quant_14/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/add�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource;^activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource;^activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource;^activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource;^activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentitydense_noise/Softmax:softmax:0;^activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_14/Minimum/ReadVariableOp<^activation_quant_14/relux/Regularizer/Square/ReadVariableOp;^activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_15/Minimum/ReadVariableOp<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp;^activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_16/Minimum/ReadVariableOp<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp;^activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_17/Minimum/ReadVariableOp<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp;^activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_18/Minimum/ReadVariableOp<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp7^batch_normalization_15/FusedBatchNormV3/ReadVariableOp9^batch_normalization_15/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1'^conv2d_noise_17/BiasAdd/ReadVariableOp+^conv2d_noise_17/convolution/ReadVariableOp'^conv2d_noise_18/BiasAdd/ReadVariableOp+^conv2d_noise_18/convolution/ReadVariableOp'^conv2d_noise_19/BiasAdd/ReadVariableOp+^conv2d_noise_19/convolution/ReadVariableOp'^conv2d_noise_20/BiasAdd/ReadVariableOp+^conv2d_noise_20/convolution/ReadVariableOp"^dense_noise/MatMul/ReadVariableOp^dense_noise/add/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::2x
:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_14/Minimum/ReadVariableOp*activation_quant_14/Minimum/ReadVariableOp2z
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp;activation_quant_14/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_15/Minimum/ReadVariableOp*activation_quant_15/Minimum/ReadVariableOp2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_16/Minimum/ReadVariableOp*activation_quant_16/Minimum/ReadVariableOp2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_17/Minimum/ReadVariableOp*activation_quant_17/Minimum/ReadVariableOp2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_18/Minimum/ReadVariableOp*activation_quant_18/Minimum/ReadVariableOp2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2p
6batch_normalization_15/FusedBatchNormV3/ReadVariableOp6batch_normalization_15/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_15/FusedBatchNormV3/ReadVariableOp_18batch_normalization_15/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12P
&conv2d_noise_17/BiasAdd/ReadVariableOp&conv2d_noise_17/BiasAdd/ReadVariableOp2X
*conv2d_noise_17/convolution/ReadVariableOp*conv2d_noise_17/convolution/ReadVariableOp2P
&conv2d_noise_18/BiasAdd/ReadVariableOp&conv2d_noise_18/BiasAdd/ReadVariableOp2X
*conv2d_noise_18/convolution/ReadVariableOp*conv2d_noise_18/convolution/ReadVariableOp2P
&conv2d_noise_19/BiasAdd/ReadVariableOp&conv2d_noise_19/BiasAdd/ReadVariableOp2X
*conv2d_noise_19/convolution/ReadVariableOp*conv2d_noise_19/convolution/ReadVariableOp2P
&conv2d_noise_20/BiasAdd/ReadVariableOp&conv2d_noise_20/BiasAdd/ReadVariableOp2X
*conv2d_noise_20/convolution/ReadVariableOp*conv2d_noise_20/convolution/ReadVariableOp2F
!dense_noise/MatMul/ReadVariableOp!dense_noise/MatMul/ReadVariableOp2@
dense_noise/add/ReadVariableOpdense_noise/add/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417905

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
i
?__inference_add_layer_call_and_return_conditional_losses_414949

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_418162H
Dactivation_quant_17_relux_regularizer_square_readvariableop_resource
identity��;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOpDactivation_quant_17_relux_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
IdentityIdentity-activation_quant_17/relux/Regularizer/add:z:0<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp
�
�
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_417057
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
4__inference_activation_quant_16_layer_call_fn_417535
x"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_4153292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
0__inference_conv2d_noise_20_layer_call_fn_417837
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_4155472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
7__inference_batch_normalization_15_layer_call_fn_417241

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_4151022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_415146
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�	
$__inference_signature_wrapper_416307
input_1
input_2"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinput_1input_2statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8**
f%R#
!__inference__wrapped_model_4143422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_415621

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
O__inference_activation_quant_14_layer_call_and_return_conditional_losses_417021
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_14/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_14/relux/Regularizer/SquareSquareCactivation_quant_14/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_14/relux/Regularizer/Square�
+activation_quant_14/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_14/relux/Regularizer/Const�
)activation_quant_14/relux/Regularizer/SumSum0activation_quant_14/relux/Regularizer/Square:y:04activation_quant_14/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/Sum�
+activation_quant_14/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_14/relux/Regularizer/mul/x�
)activation_quant_14/relux/Regularizer/mulMul4activation_quant_14/relux/Regularizer/mul/x:output:02activation_quant_14/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/mul�
+activation_quant_14/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_14/relux/Regularizer/add/x�
)activation_quant_14/relux/Regularizer/addAddV24activation_quant_14/relux/Regularizer/add/x:output:0-activation_quant_14/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_14/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp;activation_quant_14/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
7__inference_batch_normalization_17_layer_call_fn_417749

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_4154532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�	
&__inference_model_layer_call_fn_416931
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4161392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
I
2__inference_noise_injection_1_layer_call_fn_416981
x
identity�
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_4149302
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�
D
(__inference_flatten_layer_call_fn_418020

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4156672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
��
�!
A__inference_model_layer_call_and_return_conditional_losses_416663
inputs_0
inputs_17
3activation_quant_14_minimum_readvariableop_resource/
+conv2d_noise_17_abs_readvariableop_resource-
)conv2d_noise_17_readvariableop_1_resource2
.batch_normalization_15_readvariableop_resource4
0batch_normalization_15_readvariableop_1_resource1
-batch_normalization_15_assignmovingavg_4163763
/batch_normalization_15_assignmovingavg_1_4163837
3activation_quant_15_minimum_readvariableop_resource/
+conv2d_noise_18_abs_readvariableop_resource-
)conv2d_noise_18_readvariableop_1_resource2
.batch_normalization_16_readvariableop_resource4
0batch_normalization_16_readvariableop_1_resource1
-batch_normalization_16_assignmovingavg_4164403
/batch_normalization_16_assignmovingavg_1_4164477
3activation_quant_16_minimum_readvariableop_resource/
+conv2d_noise_19_abs_readvariableop_resource-
)conv2d_noise_19_readvariableop_1_resource2
.batch_normalization_17_readvariableop_resource4
0batch_normalization_17_readvariableop_1_resource1
-batch_normalization_17_assignmovingavg_4165053
/batch_normalization_17_assignmovingavg_1_4165127
3activation_quant_17_minimum_readvariableop_resource/
+conv2d_noise_20_abs_readvariableop_resource-
)conv2d_noise_20_readvariableop_1_resource2
.batch_normalization_18_readvariableop_resource4
0batch_normalization_18_readvariableop_1_resource1
-batch_normalization_18_assignmovingavg_4165693
/batch_normalization_18_assignmovingavg_1_4165767
3activation_quant_18_minimum_readvariableop_resource+
'dense_noise_abs_readvariableop_resource)
%dense_noise_readvariableop_1_resource
identity��:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_14/Minimum/ReadVariableOp�;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_15/Minimum/ReadVariableOp�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_16/Minimum/ReadVariableOp�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_17/Minimum/ReadVariableOp�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_18/Minimum/ReadVariableOp�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_15/AssignMovingAvg/ReadVariableOp�<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�%batch_normalization_15/ReadVariableOp�'batch_normalization_15/ReadVariableOp_1�:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�%batch_normalization_16/ReadVariableOp�'batch_normalization_16/ReadVariableOp_1�:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_17/AssignMovingAvg/ReadVariableOp�<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�%batch_normalization_17/ReadVariableOp�'batch_normalization_17/ReadVariableOp_1�:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_18/AssignMovingAvg/ReadVariableOp�<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp�%batch_normalization_18/ReadVariableOp�'batch_normalization_18/ReadVariableOp_1�"conv2d_noise_17/Abs/ReadVariableOp�conv2d_noise_17/ReadVariableOp� conv2d_noise_17/ReadVariableOp_1�"conv2d_noise_18/Abs/ReadVariableOp�conv2d_noise_18/ReadVariableOp� conv2d_noise_18/ReadVariableOp_1�"conv2d_noise_19/Abs/ReadVariableOp�conv2d_noise_19/ReadVariableOp� conv2d_noise_19/ReadVariableOp_1�"conv2d_noise_20/Abs/ReadVariableOp�conv2d_noise_20/ReadVariableOp� conv2d_noise_20/ReadVariableOp_1�dense_noise/Abs/ReadVariableOp�dense_noise/ReadVariableOp�dense_noise/ReadVariableOp_1f
noise_injection/ShapeShapeinputs_0*
T0*
_output_shapes
:2
noise_injection/Shape�
"noise_injection/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"noise_injection/random_normal/mean�
$noise_injection/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$noise_injection/random_normal/stddev�
2noise_injection/random_normal/RandomStandardNormalRandomStandardNormalnoise_injection/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype024
2noise_injection/random_normal/RandomStandardNormal�
!noise_injection/random_normal/mulMul;noise_injection/random_normal/RandomStandardNormal:output:0-noise_injection/random_normal/stddev:output:0*
T0*/
_output_shapes
:���������@2#
!noise_injection/random_normal/mul�
noise_injection/random_normalAdd%noise_injection/random_normal/mul:z:0+noise_injection/random_normal/mean:output:0*
T0*/
_output_shapes
:���������@2
noise_injection/random_normal�
noise_injection/addAddV2inputs_0!noise_injection/random_normal:z:0*
T0*/
_output_shapes
:���������@2
noise_injection/addj
noise_injection_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2
noise_injection_1/Shape�
$noise_injection_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$noise_injection_1/random_normal/mean�
&noise_injection_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&noise_injection_1/random_normal/stddev�
4noise_injection_1/random_normal/RandomStandardNormalRandomStandardNormal noise_injection_1/Shape:output:0*
T0*/
_output_shapes
:���������@*
dtype026
4noise_injection_1/random_normal/RandomStandardNormal�
#noise_injection_1/random_normal/mulMul=noise_injection_1/random_normal/RandomStandardNormal:output:0/noise_injection_1/random_normal/stddev:output:0*
T0*/
_output_shapes
:���������@2%
#noise_injection_1/random_normal/mul�
noise_injection_1/random_normalAdd'noise_injection_1/random_normal/mul:z:0-noise_injection_1/random_normal/mean:output:0*
T0*/
_output_shapes
:���������@2!
noise_injection_1/random_normal�
noise_injection_1/addAddV2inputs_1#noise_injection_1/random_normal:z:0*
T0*/
_output_shapes
:���������@2
noise_injection_1/add�
add/addAddV2noise_injection/add:z:0noise_injection_1/add:z:0*
T0*/
_output_shapes
:���������@2	
add/add�
*activation_quant_14/Minimum/ReadVariableOpReadVariableOp3activation_quant_14_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_14/Minimum/ReadVariableOp�
activation_quant_14/MinimumMinimumadd/add:z:02activation_quant_14/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_14/Minimum�
activation_quant_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_14/Maximum/y�
activation_quant_14/MaximumMaximumactivation_quant_14/Minimum:z:0&activation_quant_14/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_14/Maximum{
activation_quant_14/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_14/Const�
:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_14_minimum_readvariableop_resource+^activation_quant_14/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_14/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_14/Maximum:z:0"activation_quant_14/Const:output:0Bactivation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_14/FakeQuantWithMinMaxVars�
"conv2d_noise_17/Abs/ReadVariableOpReadVariableOp+conv2d_noise_17_abs_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"conv2d_noise_17/Abs/ReadVariableOp�
conv2d_noise_17/AbsAbs*conv2d_noise_17/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
conv2d_noise_17/Abs�
conv2d_noise_17/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
conv2d_noise_17/Const�
conv2d_noise_17/MaxMaxconv2d_noise_17/Abs:y:0conv2d_noise_17/Const:output:0*
T0*
_output_shapes
: 2
conv2d_noise_17/Maxs
conv2d_noise_17/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_17/mul/y�
conv2d_noise_17/mulMulconv2d_noise_17/Max:output:0conv2d_noise_17/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_17/mul�
#conv2d_noise_17/random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2%
#conv2d_noise_17/random_normal/shape�
"conv2d_noise_17/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_noise_17/random_normal/mean�
2conv2d_noise_17/random_normal/RandomStandardNormalRandomStandardNormal,conv2d_noise_17/random_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype024
2conv2d_noise_17/random_normal/RandomStandardNormal�
!conv2d_noise_17/random_normal/mulMul;conv2d_noise_17/random_normal/RandomStandardNormal:output:0conv2d_noise_17/mul:z:0*
T0*&
_output_shapes
:@@2#
!conv2d_noise_17/random_normal/mul�
conv2d_noise_17/random_normalAdd%conv2d_noise_17/random_normal/mul:z:0+conv2d_noise_17/random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
conv2d_noise_17/random_normal�
conv2d_noise_17/ReadVariableOpReadVariableOp+conv2d_noise_17_abs_readvariableop_resource#^conv2d_noise_17/Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_noise_17/ReadVariableOp�
conv2d_noise_17/addAddV2&conv2d_noise_17/ReadVariableOp:value:0!conv2d_noise_17/random_normal:z:0*
T0*&
_output_shapes
:@@2
conv2d_noise_17/addw
conv2d_noise_17/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_17/mul_1/y�
conv2d_noise_17/mul_1Mulconv2d_noise_17/Max:output:0 conv2d_noise_17/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_17/mul_1�
%conv2d_noise_17/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2'
%conv2d_noise_17/random_normal_1/shape�
$conv2d_noise_17/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv2d_noise_17/random_normal_1/mean�
4conv2d_noise_17/random_normal_1/RandomStandardNormalRandomStandardNormal.conv2d_noise_17/random_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype026
4conv2d_noise_17/random_normal_1/RandomStandardNormal�
#conv2d_noise_17/random_normal_1/mulMul=conv2d_noise_17/random_normal_1/RandomStandardNormal:output:0conv2d_noise_17/mul_1:z:0*
T0*
_output_shapes
:@2%
#conv2d_noise_17/random_normal_1/mul�
conv2d_noise_17/random_normal_1Add'conv2d_noise_17/random_normal_1/mul:z:0-conv2d_noise_17/random_normal_1/mean:output:0*
T0*
_output_shapes
:@2!
conv2d_noise_17/random_normal_1�
 conv2d_noise_17/ReadVariableOp_1ReadVariableOp)conv2d_noise_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 conv2d_noise_17/ReadVariableOp_1�
conv2d_noise_17/add_1AddV2(conv2d_noise_17/ReadVariableOp_1:value:0#conv2d_noise_17/random_normal_1:z:0*
T0*
_output_shapes
:@2
conv2d_noise_17/add_1�
conv2d_noise_17/convolutionConv2D5activation_quant_14/FakeQuantWithMinMaxVars:outputs:0conv2d_noise_17/add:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_17/convolution�
conv2d_noise_17/BiasAddBiasAdd$conv2d_noise_17/convolution:output:0conv2d_noise_17/add_1:z:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_17/BiasAdd�
#batch_normalization_15/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_15/LogicalAnd/x�
#batch_normalization_15/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_15/LogicalAnd/y�
!batch_normalization_15/LogicalAnd
LogicalAnd,batch_normalization_15/LogicalAnd/x:output:0,batch_normalization_15/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_15/LogicalAnd�
%batch_normalization_15/ReadVariableOpReadVariableOp.batch_normalization_15_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_15/ReadVariableOp�
'batch_normalization_15/ReadVariableOp_1ReadVariableOp0batch_normalization_15_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_15/ReadVariableOp_1
batch_normalization_15/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_15/Const�
batch_normalization_15/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_15/Const_1�
'batch_normalization_15/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_17/BiasAdd:output:0-batch_normalization_15/ReadVariableOp:value:0/batch_normalization_15/ReadVariableOp_1:value:0%batch_normalization_15/Const:output:0'batch_normalization_15/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2)
'batch_normalization_15/FusedBatchNormV3�
batch_normalization_15/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2 
batch_normalization_15/Const_2�
,batch_normalization_15/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_15/AssignMovingAvg/416376*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,batch_normalization_15/AssignMovingAvg/sub/x�
*batch_normalization_15/AssignMovingAvg/subSub5batch_normalization_15/AssignMovingAvg/sub/x:output:0'batch_normalization_15/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_15/AssignMovingAvg/416376*
_output_shapes
: 2,
*batch_normalization_15/AssignMovingAvg/sub�
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_15_assignmovingavg_416376*
_output_shapes
:@*
dtype027
5batch_normalization_15/AssignMovingAvg/ReadVariableOp�
,batch_normalization_15/AssignMovingAvg/sub_1Sub=batch_normalization_15/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_15/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_15/AssignMovingAvg/416376*
_output_shapes
:@2.
,batch_normalization_15/AssignMovingAvg/sub_1�
*batch_normalization_15/AssignMovingAvg/mulMul0batch_normalization_15/AssignMovingAvg/sub_1:z:0.batch_normalization_15/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_15/AssignMovingAvg/416376*
_output_shapes
:@2,
*batch_normalization_15/AssignMovingAvg/mul�
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_15_assignmovingavg_416376.batch_normalization_15/AssignMovingAvg/mul:z:06^batch_normalization_15/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_15/AssignMovingAvg/416376*
_output_shapes
 *
dtype02<
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_15/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_15/AssignMovingAvg_1/416383*
_output_shapes
: *
dtype0*
valueB
 *  �?20
.batch_normalization_15/AssignMovingAvg_1/sub/x�
,batch_normalization_15/AssignMovingAvg_1/subSub7batch_normalization_15/AssignMovingAvg_1/sub/x:output:0'batch_normalization_15/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_15/AssignMovingAvg_1/416383*
_output_shapes
: 2.
,batch_normalization_15/AssignMovingAvg_1/sub�
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_15_assignmovingavg_1_416383*
_output_shapes
:@*
dtype029
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp�
.batch_normalization_15/AssignMovingAvg_1/sub_1Sub?batch_normalization_15/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_15/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_15/AssignMovingAvg_1/416383*
_output_shapes
:@20
.batch_normalization_15/AssignMovingAvg_1/sub_1�
,batch_normalization_15/AssignMovingAvg_1/mulMul2batch_normalization_15/AssignMovingAvg_1/sub_1:z:00batch_normalization_15/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_15/AssignMovingAvg_1/416383*
_output_shapes
:@2.
,batch_normalization_15/AssignMovingAvg_1/mul�
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_15_assignmovingavg_1_4163830batch_normalization_15/AssignMovingAvg_1/mul:z:08^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_15/AssignMovingAvg_1/416383*
_output_shapes
 *
dtype02>
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp�
*activation_quant_15/Minimum/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_15/Minimum/ReadVariableOp�
activation_quant_15/MinimumMinimum+batch_normalization_15/FusedBatchNormV3:y:02activation_quant_15/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_15/Minimum�
activation_quant_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_15/Maximum/y�
activation_quant_15/MaximumMaximumactivation_quant_15/Minimum:z:0&activation_quant_15/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_15/Maximum{
activation_quant_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_15/Const�
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource+^activation_quant_15/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_15/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_15/Maximum:z:0"activation_quant_15/Const:output:0Bactivation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_15/FakeQuantWithMinMaxVars�
"conv2d_noise_18/Abs/ReadVariableOpReadVariableOp+conv2d_noise_18_abs_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"conv2d_noise_18/Abs/ReadVariableOp�
conv2d_noise_18/AbsAbs*conv2d_noise_18/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
conv2d_noise_18/Abs�
conv2d_noise_18/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
conv2d_noise_18/Const�
conv2d_noise_18/MaxMaxconv2d_noise_18/Abs:y:0conv2d_noise_18/Const:output:0*
T0*
_output_shapes
: 2
conv2d_noise_18/Maxs
conv2d_noise_18/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_18/mul/y�
conv2d_noise_18/mulMulconv2d_noise_18/Max:output:0conv2d_noise_18/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_18/mul�
#conv2d_noise_18/random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2%
#conv2d_noise_18/random_normal/shape�
"conv2d_noise_18/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_noise_18/random_normal/mean�
2conv2d_noise_18/random_normal/RandomStandardNormalRandomStandardNormal,conv2d_noise_18/random_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype024
2conv2d_noise_18/random_normal/RandomStandardNormal�
!conv2d_noise_18/random_normal/mulMul;conv2d_noise_18/random_normal/RandomStandardNormal:output:0conv2d_noise_18/mul:z:0*
T0*&
_output_shapes
:@@2#
!conv2d_noise_18/random_normal/mul�
conv2d_noise_18/random_normalAdd%conv2d_noise_18/random_normal/mul:z:0+conv2d_noise_18/random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
conv2d_noise_18/random_normal�
conv2d_noise_18/ReadVariableOpReadVariableOp+conv2d_noise_18_abs_readvariableop_resource#^conv2d_noise_18/Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_noise_18/ReadVariableOp�
conv2d_noise_18/addAddV2&conv2d_noise_18/ReadVariableOp:value:0!conv2d_noise_18/random_normal:z:0*
T0*&
_output_shapes
:@@2
conv2d_noise_18/addw
conv2d_noise_18/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_18/mul_1/y�
conv2d_noise_18/mul_1Mulconv2d_noise_18/Max:output:0 conv2d_noise_18/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_18/mul_1�
%conv2d_noise_18/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2'
%conv2d_noise_18/random_normal_1/shape�
$conv2d_noise_18/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv2d_noise_18/random_normal_1/mean�
4conv2d_noise_18/random_normal_1/RandomStandardNormalRandomStandardNormal.conv2d_noise_18/random_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype026
4conv2d_noise_18/random_normal_1/RandomStandardNormal�
#conv2d_noise_18/random_normal_1/mulMul=conv2d_noise_18/random_normal_1/RandomStandardNormal:output:0conv2d_noise_18/mul_1:z:0*
T0*
_output_shapes
:@2%
#conv2d_noise_18/random_normal_1/mul�
conv2d_noise_18/random_normal_1Add'conv2d_noise_18/random_normal_1/mul:z:0-conv2d_noise_18/random_normal_1/mean:output:0*
T0*
_output_shapes
:@2!
conv2d_noise_18/random_normal_1�
 conv2d_noise_18/ReadVariableOp_1ReadVariableOp)conv2d_noise_18_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 conv2d_noise_18/ReadVariableOp_1�
conv2d_noise_18/add_1AddV2(conv2d_noise_18/ReadVariableOp_1:value:0#conv2d_noise_18/random_normal_1:z:0*
T0*
_output_shapes
:@2
conv2d_noise_18/add_1�
conv2d_noise_18/convolutionConv2D5activation_quant_15/FakeQuantWithMinMaxVars:outputs:0conv2d_noise_18/add:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_18/convolution�
conv2d_noise_18/BiasAddBiasAdd$conv2d_noise_18/convolution:output:0conv2d_noise_18/add_1:z:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_18/BiasAdd�
#batch_normalization_16/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_16/LogicalAnd/x�
#batch_normalization_16/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_16/LogicalAnd/y�
!batch_normalization_16/LogicalAnd
LogicalAnd,batch_normalization_16/LogicalAnd/x:output:0,batch_normalization_16/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_16/LogicalAnd�
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOp�
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1
batch_normalization_16/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_16/Const�
batch_normalization_16/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_16/Const_1�
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_18/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0%batch_normalization_16/Const:output:0'batch_normalization_16/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2)
'batch_normalization_16/FusedBatchNormV3�
batch_normalization_16/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2 
batch_normalization_16/Const_2�
,batch_normalization_16/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/416440*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,batch_normalization_16/AssignMovingAvg/sub/x�
*batch_normalization_16/AssignMovingAvg/subSub5batch_normalization_16/AssignMovingAvg/sub/x:output:0'batch_normalization_16/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/416440*
_output_shapes
: 2,
*batch_normalization_16/AssignMovingAvg/sub�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_16_assignmovingavg_416440*
_output_shapes
:@*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOp�
,batch_normalization_16/AssignMovingAvg/sub_1Sub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_16/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/416440*
_output_shapes
:@2.
,batch_normalization_16/AssignMovingAvg/sub_1�
*batch_normalization_16/AssignMovingAvg/mulMul0batch_normalization_16/AssignMovingAvg/sub_1:z:0.batch_normalization_16/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/416440*
_output_shapes
:@2,
*batch_normalization_16/AssignMovingAvg/mul�
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_16_assignmovingavg_416440.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/416440*
_output_shapes
 *
dtype02<
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_16/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/416447*
_output_shapes
: *
dtype0*
valueB
 *  �?20
.batch_normalization_16/AssignMovingAvg_1/sub/x�
,batch_normalization_16/AssignMovingAvg_1/subSub7batch_normalization_16/AssignMovingAvg_1/sub/x:output:0'batch_normalization_16/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/416447*
_output_shapes
: 2.
,batch_normalization_16/AssignMovingAvg_1/sub�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_16_assignmovingavg_1_416447*
_output_shapes
:@*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�
.batch_normalization_16/AssignMovingAvg_1/sub_1Sub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_16/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/416447*
_output_shapes
:@20
.batch_normalization_16/AssignMovingAvg_1/sub_1�
,batch_normalization_16/AssignMovingAvg_1/mulMul2batch_normalization_16/AssignMovingAvg_1/sub_1:z:00batch_normalization_16/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/416447*
_output_shapes
:@2.
,batch_normalization_16/AssignMovingAvg_1/mul�
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_16_assignmovingavg_1_4164470batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/416447*
_output_shapes
 *
dtype02>
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp�
	add_1/addAddV25activation_quant_14/FakeQuantWithMinMaxVars:outputs:0+batch_normalization_16/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
	add_1/add�
*activation_quant_16/Minimum/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_16/Minimum/ReadVariableOp�
activation_quant_16/MinimumMinimumadd_1/add:z:02activation_quant_16/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_16/Minimum�
activation_quant_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_16/Maximum/y�
activation_quant_16/MaximumMaximumactivation_quant_16/Minimum:z:0&activation_quant_16/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_16/Maximum{
activation_quant_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_16/Const�
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource+^activation_quant_16/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_16/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_16/Maximum:z:0"activation_quant_16/Const:output:0Bactivation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_16/FakeQuantWithMinMaxVars�
"conv2d_noise_19/Abs/ReadVariableOpReadVariableOp+conv2d_noise_19_abs_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"conv2d_noise_19/Abs/ReadVariableOp�
conv2d_noise_19/AbsAbs*conv2d_noise_19/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
conv2d_noise_19/Abs�
conv2d_noise_19/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
conv2d_noise_19/Const�
conv2d_noise_19/MaxMaxconv2d_noise_19/Abs:y:0conv2d_noise_19/Const:output:0*
T0*
_output_shapes
: 2
conv2d_noise_19/Maxs
conv2d_noise_19/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_19/mul/y�
conv2d_noise_19/mulMulconv2d_noise_19/Max:output:0conv2d_noise_19/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_19/mul�
#conv2d_noise_19/random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2%
#conv2d_noise_19/random_normal/shape�
"conv2d_noise_19/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_noise_19/random_normal/mean�
2conv2d_noise_19/random_normal/RandomStandardNormalRandomStandardNormal,conv2d_noise_19/random_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype024
2conv2d_noise_19/random_normal/RandomStandardNormal�
!conv2d_noise_19/random_normal/mulMul;conv2d_noise_19/random_normal/RandomStandardNormal:output:0conv2d_noise_19/mul:z:0*
T0*&
_output_shapes
:@@2#
!conv2d_noise_19/random_normal/mul�
conv2d_noise_19/random_normalAdd%conv2d_noise_19/random_normal/mul:z:0+conv2d_noise_19/random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
conv2d_noise_19/random_normal�
conv2d_noise_19/ReadVariableOpReadVariableOp+conv2d_noise_19_abs_readvariableop_resource#^conv2d_noise_19/Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_noise_19/ReadVariableOp�
conv2d_noise_19/addAddV2&conv2d_noise_19/ReadVariableOp:value:0!conv2d_noise_19/random_normal:z:0*
T0*&
_output_shapes
:@@2
conv2d_noise_19/addw
conv2d_noise_19/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_19/mul_1/y�
conv2d_noise_19/mul_1Mulconv2d_noise_19/Max:output:0 conv2d_noise_19/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_19/mul_1�
%conv2d_noise_19/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2'
%conv2d_noise_19/random_normal_1/shape�
$conv2d_noise_19/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv2d_noise_19/random_normal_1/mean�
4conv2d_noise_19/random_normal_1/RandomStandardNormalRandomStandardNormal.conv2d_noise_19/random_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype026
4conv2d_noise_19/random_normal_1/RandomStandardNormal�
#conv2d_noise_19/random_normal_1/mulMul=conv2d_noise_19/random_normal_1/RandomStandardNormal:output:0conv2d_noise_19/mul_1:z:0*
T0*
_output_shapes
:@2%
#conv2d_noise_19/random_normal_1/mul�
conv2d_noise_19/random_normal_1Add'conv2d_noise_19/random_normal_1/mul:z:0-conv2d_noise_19/random_normal_1/mean:output:0*
T0*
_output_shapes
:@2!
conv2d_noise_19/random_normal_1�
 conv2d_noise_19/ReadVariableOp_1ReadVariableOp)conv2d_noise_19_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 conv2d_noise_19/ReadVariableOp_1�
conv2d_noise_19/add_1AddV2(conv2d_noise_19/ReadVariableOp_1:value:0#conv2d_noise_19/random_normal_1:z:0*
T0*
_output_shapes
:@2
conv2d_noise_19/add_1�
conv2d_noise_19/convolutionConv2D5activation_quant_16/FakeQuantWithMinMaxVars:outputs:0conv2d_noise_19/add:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_19/convolution�
conv2d_noise_19/BiasAddBiasAdd$conv2d_noise_19/convolution:output:0conv2d_noise_19/add_1:z:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_19/BiasAdd�
#batch_normalization_17/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_17/LogicalAnd/x�
#batch_normalization_17/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_17/LogicalAnd/y�
!batch_normalization_17/LogicalAnd
LogicalAnd,batch_normalization_17/LogicalAnd/x:output:0,batch_normalization_17/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_17/LogicalAnd�
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_17/ReadVariableOp�
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_17/ReadVariableOp_1
batch_normalization_17/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_17/Const�
batch_normalization_17/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_17/Const_1�
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_19/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0%batch_normalization_17/Const:output:0'batch_normalization_17/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2)
'batch_normalization_17/FusedBatchNormV3�
batch_normalization_17/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2 
batch_normalization_17/Const_2�
,batch_normalization_17/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/416505*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,batch_normalization_17/AssignMovingAvg/sub/x�
*batch_normalization_17/AssignMovingAvg/subSub5batch_normalization_17/AssignMovingAvg/sub/x:output:0'batch_normalization_17/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/416505*
_output_shapes
: 2,
*batch_normalization_17/AssignMovingAvg/sub�
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_17_assignmovingavg_416505*
_output_shapes
:@*
dtype027
5batch_normalization_17/AssignMovingAvg/ReadVariableOp�
,batch_normalization_17/AssignMovingAvg/sub_1Sub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_17/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/416505*
_output_shapes
:@2.
,batch_normalization_17/AssignMovingAvg/sub_1�
*batch_normalization_17/AssignMovingAvg/mulMul0batch_normalization_17/AssignMovingAvg/sub_1:z:0.batch_normalization_17/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/416505*
_output_shapes
:@2,
*batch_normalization_17/AssignMovingAvg/mul�
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_17_assignmovingavg_416505.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/416505*
_output_shapes
 *
dtype02<
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_17/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/416512*
_output_shapes
: *
dtype0*
valueB
 *  �?20
.batch_normalization_17/AssignMovingAvg_1/sub/x�
,batch_normalization_17/AssignMovingAvg_1/subSub7batch_normalization_17/AssignMovingAvg_1/sub/x:output:0'batch_normalization_17/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/416512*
_output_shapes
: 2.
,batch_normalization_17/AssignMovingAvg_1/sub�
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_17_assignmovingavg_1_416512*
_output_shapes
:@*
dtype029
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�
.batch_normalization_17/AssignMovingAvg_1/sub_1Sub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_17/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/416512*
_output_shapes
:@20
.batch_normalization_17/AssignMovingAvg_1/sub_1�
,batch_normalization_17/AssignMovingAvg_1/mulMul2batch_normalization_17/AssignMovingAvg_1/sub_1:z:00batch_normalization_17/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/416512*
_output_shapes
:@2.
,batch_normalization_17/AssignMovingAvg_1/mul�
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_17_assignmovingavg_1_4165120batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/416512*
_output_shapes
 *
dtype02>
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp�
*activation_quant_17/Minimum/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_17/Minimum/ReadVariableOp�
activation_quant_17/MinimumMinimum+batch_normalization_17/FusedBatchNormV3:y:02activation_quant_17/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_17/Minimum�
activation_quant_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_17/Maximum/y�
activation_quant_17/MaximumMaximumactivation_quant_17/Minimum:z:0&activation_quant_17/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_17/Maximum{
activation_quant_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_17/Const�
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource+^activation_quant_17/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_17/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_17/Maximum:z:0"activation_quant_17/Const:output:0Bactivation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_17/FakeQuantWithMinMaxVars�
"conv2d_noise_20/Abs/ReadVariableOpReadVariableOp+conv2d_noise_20_abs_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"conv2d_noise_20/Abs/ReadVariableOp�
conv2d_noise_20/AbsAbs*conv2d_noise_20/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
conv2d_noise_20/Abs�
conv2d_noise_20/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
conv2d_noise_20/Const�
conv2d_noise_20/MaxMaxconv2d_noise_20/Abs:y:0conv2d_noise_20/Const:output:0*
T0*
_output_shapes
: 2
conv2d_noise_20/Maxs
conv2d_noise_20/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_20/mul/y�
conv2d_noise_20/mulMulconv2d_noise_20/Max:output:0conv2d_noise_20/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_20/mul�
#conv2d_noise_20/random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2%
#conv2d_noise_20/random_normal/shape�
"conv2d_noise_20/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_noise_20/random_normal/mean�
2conv2d_noise_20/random_normal/RandomStandardNormalRandomStandardNormal,conv2d_noise_20/random_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype024
2conv2d_noise_20/random_normal/RandomStandardNormal�
!conv2d_noise_20/random_normal/mulMul;conv2d_noise_20/random_normal/RandomStandardNormal:output:0conv2d_noise_20/mul:z:0*
T0*&
_output_shapes
:@@2#
!conv2d_noise_20/random_normal/mul�
conv2d_noise_20/random_normalAdd%conv2d_noise_20/random_normal/mul:z:0+conv2d_noise_20/random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
conv2d_noise_20/random_normal�
conv2d_noise_20/ReadVariableOpReadVariableOp+conv2d_noise_20_abs_readvariableop_resource#^conv2d_noise_20/Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_noise_20/ReadVariableOp�
conv2d_noise_20/addAddV2&conv2d_noise_20/ReadVariableOp:value:0!conv2d_noise_20/random_normal:z:0*
T0*&
_output_shapes
:@@2
conv2d_noise_20/addw
conv2d_noise_20/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_20/mul_1/y�
conv2d_noise_20/mul_1Mulconv2d_noise_20/Max:output:0 conv2d_noise_20/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_20/mul_1�
%conv2d_noise_20/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2'
%conv2d_noise_20/random_normal_1/shape�
$conv2d_noise_20/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv2d_noise_20/random_normal_1/mean�
4conv2d_noise_20/random_normal_1/RandomStandardNormalRandomStandardNormal.conv2d_noise_20/random_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype026
4conv2d_noise_20/random_normal_1/RandomStandardNormal�
#conv2d_noise_20/random_normal_1/mulMul=conv2d_noise_20/random_normal_1/RandomStandardNormal:output:0conv2d_noise_20/mul_1:z:0*
T0*
_output_shapes
:@2%
#conv2d_noise_20/random_normal_1/mul�
conv2d_noise_20/random_normal_1Add'conv2d_noise_20/random_normal_1/mul:z:0-conv2d_noise_20/random_normal_1/mean:output:0*
T0*
_output_shapes
:@2!
conv2d_noise_20/random_normal_1�
 conv2d_noise_20/ReadVariableOp_1ReadVariableOp)conv2d_noise_20_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 conv2d_noise_20/ReadVariableOp_1�
conv2d_noise_20/add_1AddV2(conv2d_noise_20/ReadVariableOp_1:value:0#conv2d_noise_20/random_normal_1:z:0*
T0*
_output_shapes
:@2
conv2d_noise_20/add_1�
conv2d_noise_20/convolutionConv2D5activation_quant_17/FakeQuantWithMinMaxVars:outputs:0conv2d_noise_20/add:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_20/convolution�
conv2d_noise_20/BiasAddBiasAdd$conv2d_noise_20/convolution:output:0conv2d_noise_20/add_1:z:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_20/BiasAdd�
#batch_normalization_18/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_18/LogicalAnd/x�
#batch_normalization_18/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_18/LogicalAnd/y�
!batch_normalization_18/LogicalAnd
LogicalAnd,batch_normalization_18/LogicalAnd/x:output:0,batch_normalization_18/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_18/LogicalAnd�
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_18/ReadVariableOp�
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_18/ReadVariableOp_1
batch_normalization_18/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_18/Const�
batch_normalization_18/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_18/Const_1�
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_20/BiasAdd:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0%batch_normalization_18/Const:output:0'batch_normalization_18/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2)
'batch_normalization_18/FusedBatchNormV3�
batch_normalization_18/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2 
batch_normalization_18/Const_2�
,batch_normalization_18/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/416569*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,batch_normalization_18/AssignMovingAvg/sub/x�
*batch_normalization_18/AssignMovingAvg/subSub5batch_normalization_18/AssignMovingAvg/sub/x:output:0'batch_normalization_18/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/416569*
_output_shapes
: 2,
*batch_normalization_18/AssignMovingAvg/sub�
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_18_assignmovingavg_416569*
_output_shapes
:@*
dtype027
5batch_normalization_18/AssignMovingAvg/ReadVariableOp�
,batch_normalization_18/AssignMovingAvg/sub_1Sub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_18/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/416569*
_output_shapes
:@2.
,batch_normalization_18/AssignMovingAvg/sub_1�
*batch_normalization_18/AssignMovingAvg/mulMul0batch_normalization_18/AssignMovingAvg/sub_1:z:0.batch_normalization_18/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/416569*
_output_shapes
:@2,
*batch_normalization_18/AssignMovingAvg/mul�
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_18_assignmovingavg_416569.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/416569*
_output_shapes
 *
dtype02<
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_18/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/416576*
_output_shapes
: *
dtype0*
valueB
 *  �?20
.batch_normalization_18/AssignMovingAvg_1/sub/x�
,batch_normalization_18/AssignMovingAvg_1/subSub7batch_normalization_18/AssignMovingAvg_1/sub/x:output:0'batch_normalization_18/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/416576*
_output_shapes
: 2.
,batch_normalization_18/AssignMovingAvg_1/sub�
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_18_assignmovingavg_1_416576*
_output_shapes
:@*
dtype029
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp�
.batch_normalization_18/AssignMovingAvg_1/sub_1Sub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_18/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/416576*
_output_shapes
:@20
.batch_normalization_18/AssignMovingAvg_1/sub_1�
,batch_normalization_18/AssignMovingAvg_1/mulMul2batch_normalization_18/AssignMovingAvg_1/sub_1:z:00batch_normalization_18/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/416576*
_output_shapes
:@2.
,batch_normalization_18/AssignMovingAvg_1/mul�
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_18_assignmovingavg_1_4165760batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/416576*
_output_shapes
 *
dtype02>
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp�
	add_2/addAddV25activation_quant_16/FakeQuantWithMinMaxVars:outputs:0+batch_normalization_18/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
	add_2/add�
average_pooling2d/AvgPoolAvgPooladd_2/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
flatten/Const�
flatten/ReshapeReshape"average_pooling2d/AvgPool:output:0flatten/Const:output:0*
T0*'
_output_shapes
:���������@2
flatten/Reshape�
*activation_quant_18/Minimum/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_18/Minimum/ReadVariableOp�
activation_quant_18/MinimumMinimumflatten/Reshape:output:02activation_quant_18/Minimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
activation_quant_18/Minimum�
activation_quant_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_18/Maximum/y�
activation_quant_18/MaximumMaximumactivation_quant_18/Minimum:z:0&activation_quant_18/Maximum/y:output:0*
T0*'
_output_shapes
:���������@2
activation_quant_18/Maximum{
activation_quant_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_18/Const�
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource+^activation_quant_18/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_18/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_18/Maximum:z:0"activation_quant_18/Const:output:0Bactivation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits2-
+activation_quant_18/FakeQuantWithMinMaxVars�
dense_noise/Abs/ReadVariableOpReadVariableOp'dense_noise_abs_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_noise/Abs/ReadVariableOpz
dense_noise/AbsAbs&dense_noise/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
dense_noise/Absw
dense_noise/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_noise/Const{
dense_noise/MaxMaxdense_noise/Abs:y:0dense_noise/Const:output:0*
T0*
_output_shapes
: 2
dense_noise/Maxk
dense_noise/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dense_noise/mul/y�
dense_noise/mulMuldense_noise/Max:output:0dense_noise/mul/y:output:0*
T0*
_output_shapes
: 2
dense_noise/mul�
dense_noise/random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   
   2!
dense_noise/random_normal/shape�
dense_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_noise/random_normal/mean�
.dense_noise/random_normal/RandomStandardNormalRandomStandardNormal(dense_noise/random_normal/shape:output:0*
T0*
_output_shapes

:@
*
dtype020
.dense_noise/random_normal/RandomStandardNormal�
dense_noise/random_normal/mulMul7dense_noise/random_normal/RandomStandardNormal:output:0dense_noise/mul:z:0*
T0*
_output_shapes

:@
2
dense_noise/random_normal/mul�
dense_noise/random_normalAdd!dense_noise/random_normal/mul:z:0'dense_noise/random_normal/mean:output:0*
T0*
_output_shapes

:@
2
dense_noise/random_normal�
dense_noise/ReadVariableOpReadVariableOp'dense_noise_abs_readvariableop_resource^dense_noise/Abs/ReadVariableOp*
_output_shapes

:@
*
dtype02
dense_noise/ReadVariableOp�
dense_noise/addAddV2"dense_noise/ReadVariableOp:value:0dense_noise/random_normal:z:0*
T0*
_output_shapes

:@
2
dense_noise/addo
dense_noise/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dense_noise/mul_1/y�
dense_noise/mul_1Muldense_noise/Max:output:0dense_noise/mul_1/y:output:0*
T0*
_output_shapes
: 2
dense_noise/mul_1�
!dense_noise/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
2#
!dense_noise/random_normal_1/shape�
 dense_noise/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_noise/random_normal_1/mean�
0dense_noise/random_normal_1/RandomStandardNormalRandomStandardNormal*dense_noise/random_normal_1/shape:output:0*
T0*
_output_shapes
:
*
dtype022
0dense_noise/random_normal_1/RandomStandardNormal�
dense_noise/random_normal_1/mulMul9dense_noise/random_normal_1/RandomStandardNormal:output:0dense_noise/mul_1:z:0*
T0*
_output_shapes
:
2!
dense_noise/random_normal_1/mul�
dense_noise/random_normal_1Add#dense_noise/random_normal_1/mul:z:0)dense_noise/random_normal_1/mean:output:0*
T0*
_output_shapes
:
2
dense_noise/random_normal_1�
dense_noise/ReadVariableOp_1ReadVariableOp%dense_noise_readvariableop_1_resource*
_output_shapes
:
*
dtype02
dense_noise/ReadVariableOp_1�
dense_noise/add_1AddV2$dense_noise/ReadVariableOp_1:value:0dense_noise/random_normal_1:z:0*
T0*
_output_shapes
:
2
dense_noise/add_1�
dense_noise/MatMulMatMul5activation_quant_18/FakeQuantWithMinMaxVars:outputs:0dense_noise/add:z:0*
T0*'
_output_shapes
:���������
2
dense_noise/MatMul�
dense_noise/add_2AddV2dense_noise/MatMul:product:0dense_noise/add_1:z:0*
T0*'
_output_shapes
:���������
2
dense_noise/add_2~
dense_noise/SoftmaxSoftmaxdense_noise/add_2:z:0*
T0*'
_output_shapes
:���������
2
dense_noise/Softmax�
;activation_quant_14/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_14_minimum_readvariableop_resource;^activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_14/relux/Regularizer/SquareSquareCactivation_quant_14/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_14/relux/Regularizer/Square�
+activation_quant_14/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_14/relux/Regularizer/Const�
)activation_quant_14/relux/Regularizer/SumSum0activation_quant_14/relux/Regularizer/Square:y:04activation_quant_14/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/Sum�
+activation_quant_14/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_14/relux/Regularizer/mul/x�
)activation_quant_14/relux/Regularizer/mulMul4activation_quant_14/relux/Regularizer/mul/x:output:02activation_quant_14/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/mul�
+activation_quant_14/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_14/relux/Regularizer/add/x�
)activation_quant_14/relux/Regularizer/addAddV24activation_quant_14/relux/Regularizer/add/x:output:0-activation_quant_14/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/add�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource;^activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource;^activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource;^activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource;^activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentitydense_noise/Softmax:softmax:0;^activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_14/Minimum/ReadVariableOp<^activation_quant_14/relux/Regularizer/Square/ReadVariableOp;^activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_15/Minimum/ReadVariableOp<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp;^activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_16/Minimum/ReadVariableOp<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp;^activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_17/Minimum/ReadVariableOp<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp;^activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_18/Minimum/ReadVariableOp<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp;^batch_normalization_15/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_15/AssignMovingAvg/ReadVariableOp=^batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_15/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_15/ReadVariableOp(^batch_normalization_15/ReadVariableOp_1;^batch_normalization_16/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_16/AssignMovingAvg/ReadVariableOp=^batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1;^batch_normalization_17/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_17/AssignMovingAvg/ReadVariableOp=^batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1;^batch_normalization_18/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_18/AssignMovingAvg/ReadVariableOp=^batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1#^conv2d_noise_17/Abs/ReadVariableOp^conv2d_noise_17/ReadVariableOp!^conv2d_noise_17/ReadVariableOp_1#^conv2d_noise_18/Abs/ReadVariableOp^conv2d_noise_18/ReadVariableOp!^conv2d_noise_18/ReadVariableOp_1#^conv2d_noise_19/Abs/ReadVariableOp^conv2d_noise_19/ReadVariableOp!^conv2d_noise_19/ReadVariableOp_1#^conv2d_noise_20/Abs/ReadVariableOp^conv2d_noise_20/ReadVariableOp!^conv2d_noise_20/ReadVariableOp_1^dense_noise/Abs/ReadVariableOp^dense_noise/ReadVariableOp^dense_noise/ReadVariableOp_1*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::2x
:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_14/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_14/Minimum/ReadVariableOp*activation_quant_14/Minimum/ReadVariableOp2z
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp;activation_quant_14/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_15/Minimum/ReadVariableOp*activation_quant_15/Minimum/ReadVariableOp2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_16/Minimum/ReadVariableOp*activation_quant_16/Minimum/ReadVariableOp2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_17/Minimum/ReadVariableOp*activation_quant_17/Minimum/ReadVariableOp2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_18/Minimum/ReadVariableOp*activation_quant_18/Minimum/ReadVariableOp2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2x
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp:batch_normalization_15/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_15/AssignMovingAvg/ReadVariableOp5batch_normalization_15/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_15/ReadVariableOp%batch_normalization_15/ReadVariableOp2R
'batch_normalization_15/ReadVariableOp_1'batch_normalization_15/ReadVariableOp_12x
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12x
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12x
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_18/AssignMovingAvg/ReadVariableOp5batch_normalization_18/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12H
"conv2d_noise_17/Abs/ReadVariableOp"conv2d_noise_17/Abs/ReadVariableOp2@
conv2d_noise_17/ReadVariableOpconv2d_noise_17/ReadVariableOp2D
 conv2d_noise_17/ReadVariableOp_1 conv2d_noise_17/ReadVariableOp_12H
"conv2d_noise_18/Abs/ReadVariableOp"conv2d_noise_18/Abs/ReadVariableOp2@
conv2d_noise_18/ReadVariableOpconv2d_noise_18/ReadVariableOp2D
 conv2d_noise_18/ReadVariableOp_1 conv2d_noise_18/ReadVariableOp_12H
"conv2d_noise_19/Abs/ReadVariableOp"conv2d_noise_19/Abs/ReadVariableOp2@
conv2d_noise_19/ReadVariableOpconv2d_noise_19/ReadVariableOp2D
 conv2d_noise_19/ReadVariableOp_1 conv2d_noise_19/ReadVariableOp_12H
"conv2d_noise_20/Abs/ReadVariableOp"conv2d_noise_20/Abs/ReadVariableOp2@
conv2d_noise_20/ReadVariableOpconv2d_noise_20/ReadVariableOp2D
 conv2d_noise_20/ReadVariableOp_1 conv2d_noise_20/ReadVariableOp_12@
dense_noise/Abs/ReadVariableOpdense_noise/Abs/ReadVariableOp28
dense_noise/ReadVariableOpdense_noise/ReadVariableOp2<
dense_noise/ReadVariableOp_1dense_noise/ReadVariableOp_1:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
m
A__inference_add_1_layer_call_and_return_conditional_losses_417495
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417731

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
4__inference_activation_quant_14_layer_call_fn_417027
x"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_14_layer_call_and_return_conditional_losses_4149782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_415431

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_415416
assignmovingavg_1_415423
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/415416*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/415416*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_415416*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/415416*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/415416*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_415416AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/415416*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/415423*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415423*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_415423*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415423*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415423*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_415423AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/415423*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_417529
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_415329
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
b
K__inference_noise_injection_layer_call_and_return_conditional_losses_414902
x
identity]
IdentityIdentityx*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�
�
O__inference_activation_quant_14_layer_call_and_return_conditional_losses_414978
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_14/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_14/relux/Regularizer/SquareSquareCactivation_quant_14/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_14/relux/Regularizer/Square�
+activation_quant_14/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_14/relux/Regularizer/Const�
)activation_quant_14/relux/Regularizer/SumSum0activation_quant_14/relux/Regularizer/Square:y:04activation_quant_14/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/Sum�
+activation_quant_14/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_14/relux/Regularizer/mul/x�
)activation_quant_14/relux/Regularizer/mulMul4activation_quant_14/relux/Regularizer/mul/x:output:02activation_quant_14/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/mul�
+activation_quant_14/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_14/relux/Regularizer/add/x�
)activation_quant_14/relux/Regularizer/addAddV24activation_quant_14/relux/Regularizer/add/x:output:0-activation_quant_14/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_14/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp;activation_quant_14/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_417269
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417657

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
G__inference_dense_noise_layer_call_and_return_conditional_losses_415736
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*
_output_shapes

:@
*
dtype02
Abs/ReadVariableOpV
AbsAbsAbs/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
Abs_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   
   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@
*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*
_output_shapes

:@
2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@
2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*
_output_shapes

:@
*
dtype02
ReadVariableOpg
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*
_output_shapes

:@
2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:
*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:
2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:
2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:
2
add_1X
MatMulMatMulxadd:z:0*
T0*'
_output_shapes
:���������
2
MatMulf
add_2AddV2MatMul:product:0	add_1:z:0*
T0*'
_output_shapes
:���������
2
add_2Z
SoftmaxSoftmax	add_2:z:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
7__inference_batch_normalization_18_layer_call_fn_417923

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_4148632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�	
&__inference_model_layer_call_fn_416173
input_1
input_2"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32
identity��StatefulPartitionedCall�

StatefulPartitionedCallStatefulPartitionedCallinput_1input_2statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4161392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
__inference_loss_fn_2_418149H
Dactivation_quant_16_relux_regularizer_square_readvariableop_resource
identity��;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOpDactivation_quant_16_relux_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
IdentityIdentity-activation_quant_16/relux/Regularizer/add:z:0<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp
�
�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_415186
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_415080

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_415065
assignmovingavg_1_415072
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/415065*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/415065*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_415065*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/415065*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/415065*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_415065AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/415065*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/415072*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415072*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_415072*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415072*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/415072*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_415072AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/415072*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_414599

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
i
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_414876

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_415270

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_15_layer_call_fn_417158

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_4144362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_417823
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�	
g
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_416967
x
identity�?
ShapeShapex*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������@2
random_normalc
addAddV2xrandom_normal:z:0*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�
k
?__inference_add_layer_call_and_return_conditional_losses_416987
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�$
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417449

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_417434
assignmovingavg_1_417441
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/417434*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/417434*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_417434*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/417434*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/417434*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_417434AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/417434*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/417441*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417441*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_417441*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417441*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417441*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_417441AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/417441*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417635

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_417620
assignmovingavg_1_417627
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/417620*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/417620*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_417620*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/417620*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/417620*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_417620AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/417620*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/417627*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417627*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_417627*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417627*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417627*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_417627AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/417627*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_417305
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
k
A__inference_add_2_layer_call_and_return_conditional_losses_415651

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_414700

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_414685
assignmovingavg_1_414692
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/414685*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/414685*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_414685*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/414685*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/414685*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_414685AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/414685*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/414692*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414692*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_414692*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414692*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414692*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_414692AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/414692*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
b
K__inference_noise_injection_layer_call_and_return_conditional_losses_416946
x
identity]
IdentityIdentityx*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�	
e
K__inference_noise_injection_layer_call_and_return_conditional_losses_416942
x
identity�?
ShapeShapex*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*/
_output_shapes
:���������@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*/
_output_shapes
:���������@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*/
_output_shapes
:���������@2
random_normalc
addAddV2xrandom_normal:z:0*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�
�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_415537
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
d
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_416971
x
identity]
IdentityIdentityx*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417201

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_417186
assignmovingavg_1_417193
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/417186*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/417186*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_417186*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/417186*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/417186*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_417186AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/417186*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/417193*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417193*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_417193*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417193*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/417193*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_417193AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/417193*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_414832

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_414817
assignmovingavg_1_414824
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/414817*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/414817*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_414817*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/414817*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/414817*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_414817AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/414817*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/414824*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414824*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_414824*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414824*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414824*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_414824AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/414824*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_16_layer_call_fn_417406

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_4152482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_415547
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
P
$__inference_add_layer_call_fn_416993
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4149492
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
m
A__inference_add_2_layer_call_and_return_conditional_losses_418003
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�$
�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_414436

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_414421
assignmovingavg_1_414428
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/414421*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/414421*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_414421*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/414421*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/414421*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_414421AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/414421*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/414428*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414428*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_414428*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414428*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/414428*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_414428AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/414428*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
��
�2
"__inference__traced_restore_418716
file_prefix.
*assignvariableop_activation_quant_14_relux-
)assignvariableop_1_conv2d_noise_17_kernel+
'assignvariableop_2_conv2d_noise_17_bias3
/assignvariableop_3_batch_normalization_15_gamma2
.assignvariableop_4_batch_normalization_15_beta9
5assignvariableop_5_batch_normalization_15_moving_mean=
9assignvariableop_6_batch_normalization_15_moving_variance0
,assignvariableop_7_activation_quant_15_relux-
)assignvariableop_8_conv2d_noise_18_kernel+
'assignvariableop_9_conv2d_noise_18_bias4
0assignvariableop_10_batch_normalization_16_gamma3
/assignvariableop_11_batch_normalization_16_beta:
6assignvariableop_12_batch_normalization_16_moving_mean>
:assignvariableop_13_batch_normalization_16_moving_variance1
-assignvariableop_14_activation_quant_16_relux.
*assignvariableop_15_conv2d_noise_19_kernel,
(assignvariableop_16_conv2d_noise_19_bias4
0assignvariableop_17_batch_normalization_17_gamma3
/assignvariableop_18_batch_normalization_17_beta:
6assignvariableop_19_batch_normalization_17_moving_mean>
:assignvariableop_20_batch_normalization_17_moving_variance1
-assignvariableop_21_activation_quant_17_relux.
*assignvariableop_22_conv2d_noise_20_kernel,
(assignvariableop_23_conv2d_noise_20_bias4
0assignvariableop_24_batch_normalization_18_gamma3
/assignvariableop_25_batch_normalization_18_beta:
6assignvariableop_26_batch_normalization_18_moving_mean>
:assignvariableop_27_batch_normalization_18_moving_variance1
-assignvariableop_28_activation_quant_18_relux*
&assignvariableop_29_dense_noise_kernel(
$assignvariableop_30_dense_noise_bias!
assignvariableop_31_adam_iter#
assignvariableop_32_adam_beta_1#
assignvariableop_33_adam_beta_2"
assignvariableop_34_adam_decay*
&assignvariableop_35_adam_learning_rate
assignvariableop_36_total
assignvariableop_37_count8
4assignvariableop_38_adam_activation_quant_14_relux_m5
1assignvariableop_39_adam_conv2d_noise_17_kernel_m3
/assignvariableop_40_adam_conv2d_noise_17_bias_m;
7assignvariableop_41_adam_batch_normalization_15_gamma_m:
6assignvariableop_42_adam_batch_normalization_15_beta_m8
4assignvariableop_43_adam_activation_quant_15_relux_m5
1assignvariableop_44_adam_conv2d_noise_18_kernel_m3
/assignvariableop_45_adam_conv2d_noise_18_bias_m;
7assignvariableop_46_adam_batch_normalization_16_gamma_m:
6assignvariableop_47_adam_batch_normalization_16_beta_m8
4assignvariableop_48_adam_activation_quant_16_relux_m5
1assignvariableop_49_adam_conv2d_noise_19_kernel_m3
/assignvariableop_50_adam_conv2d_noise_19_bias_m;
7assignvariableop_51_adam_batch_normalization_17_gamma_m:
6assignvariableop_52_adam_batch_normalization_17_beta_m8
4assignvariableop_53_adam_activation_quant_17_relux_m5
1assignvariableop_54_adam_conv2d_noise_20_kernel_m3
/assignvariableop_55_adam_conv2d_noise_20_bias_m;
7assignvariableop_56_adam_batch_normalization_18_gamma_m:
6assignvariableop_57_adam_batch_normalization_18_beta_m8
4assignvariableop_58_adam_activation_quant_18_relux_m1
-assignvariableop_59_adam_dense_noise_kernel_m/
+assignvariableop_60_adam_dense_noise_bias_m8
4assignvariableop_61_adam_activation_quant_14_relux_v5
1assignvariableop_62_adam_conv2d_noise_17_kernel_v3
/assignvariableop_63_adam_conv2d_noise_17_bias_v;
7assignvariableop_64_adam_batch_normalization_15_gamma_v:
6assignvariableop_65_adam_batch_normalization_15_beta_v8
4assignvariableop_66_adam_activation_quant_15_relux_v5
1assignvariableop_67_adam_conv2d_noise_18_kernel_v3
/assignvariableop_68_adam_conv2d_noise_18_bias_v;
7assignvariableop_69_adam_batch_normalization_16_gamma_v:
6assignvariableop_70_adam_batch_normalization_16_beta_v8
4assignvariableop_71_adam_activation_quant_16_relux_v5
1assignvariableop_72_adam_conv2d_noise_19_kernel_v3
/assignvariableop_73_adam_conv2d_noise_19_bias_v;
7assignvariableop_74_adam_batch_normalization_17_gamma_v:
6assignvariableop_75_adam_batch_normalization_17_beta_v8
4assignvariableop_76_adam_activation_quant_17_relux_v5
1assignvariableop_77_adam_conv2d_noise_20_kernel_v3
/assignvariableop_78_adam_conv2d_noise_20_bias_v;
7assignvariableop_79_adam_batch_normalization_18_gamma_v:
6assignvariableop_80_adam_batch_normalization_18_beta_v8
4assignvariableop_81_adam_activation_quant_18_relux_v1
-assignvariableop_82_adam_dense_noise_kernel_v/
+assignvariableop_83_adam_dense_noise_bias_v
identity_85��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_9�	RestoreV2�RestoreV2_1�/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�.
value�.B�.TB5layer_with_weights-0/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/relux/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/relux/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�
value�B�TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*b
dtypesX
V2T	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp*assignvariableop_activation_quant_14_reluxIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp)assignvariableop_1_conv2d_noise_17_kernelIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp'assignvariableop_2_conv2d_noise_17_biasIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_15_gammaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_15_betaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp5assignvariableop_5_batch_normalization_15_moving_meanIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp9assignvariableop_6_batch_normalization_15_moving_varianceIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp,assignvariableop_7_activation_quant_15_reluxIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_conv2d_noise_18_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp'assignvariableop_9_conv2d_noise_18_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_16_gammaIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_16_betaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_16_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_16_moving_varianceIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_activation_quant_16_reluxIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_conv2d_noise_19_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_conv2d_noise_19_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp0assignvariableop_17_batch_normalization_17_gammaIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_17_betaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp6assignvariableop_19_batch_normalization_17_moving_meanIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp:assignvariableop_20_batch_normalization_17_moving_varianceIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp-assignvariableop_21_activation_quant_17_reluxIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_conv2d_noise_20_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_conv2d_noise_20_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_18_gammaIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp/assignvariableop_25_batch_normalization_18_betaIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp6assignvariableop_26_batch_normalization_18_moving_meanIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp:assignvariableop_27_batch_normalization_18_moving_varianceIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp-assignvariableop_28_activation_quant_18_reluxIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp&assignvariableop_29_dense_noise_kernelIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp$assignvariableop_30_dense_noise_biasIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0	*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_iterIdentity_31:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_1Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_2Identity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_decayIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp&assignvariableop_35_adam_learning_rateIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOpassignvariableop_36_totalIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOpassignvariableop_37_countIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp4assignvariableop_38_adam_activation_quant_14_relux_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp1assignvariableop_39_adam_conv2d_noise_17_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp/assignvariableop_40_adam_conv2d_noise_17_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_15_gamma_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_15_beta_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_activation_quant_15_relux_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp1assignvariableop_44_adam_conv2d_noise_18_kernel_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp/assignvariableop_45_adam_conv2d_noise_18_bias_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp7assignvariableop_46_adam_batch_normalization_16_gamma_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_16_beta_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp4assignvariableop_48_adam_activation_quant_16_relux_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp1assignvariableop_49_adam_conv2d_noise_19_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp/assignvariableop_50_adam_conv2d_noise_19_bias_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp7assignvariableop_51_adam_batch_normalization_17_gamma_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_batch_normalization_17_beta_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_activation_quant_17_relux_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp1assignvariableop_54_adam_conv2d_noise_20_kernel_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp/assignvariableop_55_adam_conv2d_noise_20_bias_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp7assignvariableop_56_adam_batch_normalization_18_gamma_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_18_beta_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp4assignvariableop_58_adam_activation_quant_18_relux_mIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp-assignvariableop_59_adam_dense_noise_kernel_mIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp+assignvariableop_60_adam_dense_noise_bias_mIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp4assignvariableop_61_adam_activation_quant_14_relux_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp1assignvariableop_62_adam_conv2d_noise_17_kernel_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp/assignvariableop_63_adam_conv2d_noise_17_bias_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp7assignvariableop_64_adam_batch_normalization_15_gamma_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_15_beta_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp4assignvariableop_66_adam_activation_quant_15_relux_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp1assignvariableop_67_adam_conv2d_noise_18_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp/assignvariableop_68_adam_conv2d_noise_18_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_batch_normalization_16_gamma_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_16_beta_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp4assignvariableop_71_adam_activation_quant_16_relux_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp1assignvariableop_72_adam_conv2d_noise_19_kernel_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp/assignvariableop_73_adam_conv2d_noise_19_bias_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adam_batch_normalization_17_gamma_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp6assignvariableop_75_adam_batch_normalization_17_beta_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp4assignvariableop_76_adam_activation_quant_17_relux_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp1assignvariableop_77_adam_conv2d_noise_20_kernel_vIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp/assignvariableop_78_adam_conv2d_noise_20_bias_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_18_gamma_vIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_18_beta_vIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80_
Identity_81IdentityRestoreV2:tensors:81*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp4assignvariableop_81_adam_activation_quant_18_relux_vIdentity_81:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_81_
Identity_82IdentityRestoreV2:tensors:82*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp-assignvariableop_82_adam_dense_noise_kernel_vIdentity_82:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_82_
Identity_83IdentityRestoreV2:tensors:83*
T0*
_output_shapes
:2
Identity_83�
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_dense_noise_bias_vIdentity_83:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_83�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_84Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_84�
Identity_85IdentityIdentity_84:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_85"#
identity_85Identity_85:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
��
�
A__inference_model_layer_call_and_return_conditional_losses_415807
input_1
input_26
2activation_quant_14_statefulpartitionedcall_args_12
.conv2d_noise_17_statefulpartitionedcall_args_12
.conv2d_noise_17_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_19
5batch_normalization_15_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_39
5batch_normalization_15_statefulpartitionedcall_args_46
2activation_quant_15_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_46
2activation_quant_16_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_46
2activation_quant_17_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_19
5batch_normalization_18_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_39
5batch_normalization_18_statefulpartitionedcall_args_46
2activation_quant_18_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_2
identity��+activation_quant_14/StatefulPartitionedCall�;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�+activation_quant_15/StatefulPartitionedCall�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�+activation_quant_16/StatefulPartitionedCall�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�+activation_quant_17/StatefulPartitionedCall�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�+activation_quant_18/StatefulPartitionedCall�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�'conv2d_noise_17/StatefulPartitionedCall�'conv2d_noise_18/StatefulPartitionedCall�'conv2d_noise_19/StatefulPartitionedCall�'conv2d_noise_20/StatefulPartitionedCall�#dense_noise/StatefulPartitionedCall�'noise_injection/StatefulPartitionedCall�)noise_injection_1/StatefulPartitionedCall�
'noise_injection/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_noise_injection_layer_call_and_return_conditional_losses_4148982)
'noise_injection/StatefulPartitionedCall�
)noise_injection_1/StatefulPartitionedCallStatefulPartitionedCallinput_2(^noise_injection/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_4149262+
)noise_injection_1/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall0noise_injection/StatefulPartitionedCall:output:02noise_injection_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4149492
add/PartitionedCall�
+activation_quant_14/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:02activation_quant_14_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_14_layer_call_and_return_conditional_losses_4149782-
+activation_quant_14/StatefulPartitionedCall�
'conv2d_noise_17/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_14/StatefulPartitionedCall:output:0.conv2d_noise_17_statefulpartitionedcall_args_1.conv2d_noise_17_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_4150182)
'conv2d_noise_17/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_17/StatefulPartitionedCall:output:05batch_normalization_15_statefulpartitionedcall_args_15batch_normalization_15_statefulpartitionedcall_args_25batch_normalization_15_statefulpartitionedcall_args_35batch_normalization_15_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41508020
.batch_normalization_15/StatefulPartitionedCall�
+activation_quant_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:02activation_quant_15_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_4151462-
+activation_quant_15/StatefulPartitionedCall�
'conv2d_noise_18/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_15/StatefulPartitionedCall:output:0.conv2d_noise_18_statefulpartitionedcall_args_1.conv2d_noise_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_4151862)
'conv2d_noise_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_41524820
.batch_normalization_16/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall4activation_quant_14/StatefulPartitionedCall:output:07batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4153002
add_1/PartitionedCall�
+activation_quant_16/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:02activation_quant_16_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_4153292-
+activation_quant_16/StatefulPartitionedCall�
'conv2d_noise_19/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:0.conv2d_noise_19_statefulpartitionedcall_args_1.conv2d_noise_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_4153692)
'conv2d_noise_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41543120
.batch_normalization_17/StatefulPartitionedCall�
+activation_quant_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:02activation_quant_17_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_4154972-
+activation_quant_17/StatefulPartitionedCall�
'conv2d_noise_20/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_17/StatefulPartitionedCall:output:0.conv2d_noise_20_statefulpartitionedcall_args_1.conv2d_noise_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_4155372)
'conv2d_noise_20/StatefulPartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_20/StatefulPartitionedCall:output:05batch_normalization_18_statefulpartitionedcall_args_15batch_normalization_18_statefulpartitionedcall_args_25batch_normalization_18_statefulpartitionedcall_args_35batch_normalization_18_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_41559920
.batch_normalization_18/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:07batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4156512
add_2/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_4148762#
!average_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4156672
flatten/PartitionedCall�
+activation_quant_18/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:02activation_quant_18_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_4156952-
+activation_quant_18/StatefulPartitionedCall�
#dense_noise/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_18/StatefulPartitionedCall:output:0*dense_noise_statefulpartitionedcall_args_1*dense_noise_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_4157362%
#dense_noise/StatefulPartitionedCall�
;activation_quant_14/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_14_statefulpartitionedcall_args_1,^activation_quant_14/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_14/relux/Regularizer/SquareSquareCactivation_quant_14/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_14/relux/Regularizer/Square�
+activation_quant_14/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_14/relux/Regularizer/Const�
)activation_quant_14/relux/Regularizer/SumSum0activation_quant_14/relux/Regularizer/Square:y:04activation_quant_14/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/Sum�
+activation_quant_14/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_14/relux/Regularizer/mul/x�
)activation_quant_14/relux/Regularizer/mulMul4activation_quant_14/relux/Regularizer/mul/x:output:02activation_quant_14/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/mul�
+activation_quant_14/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_14/relux/Regularizer/add/x�
)activation_quant_14/relux/Regularizer/addAddV24activation_quant_14/relux/Regularizer/add/x:output:0-activation_quant_14/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/add�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_15_statefulpartitionedcall_args_1,^activation_quant_15/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_16_statefulpartitionedcall_args_1,^activation_quant_16/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_17_statefulpartitionedcall_args_1,^activation_quant_17/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_18_statefulpartitionedcall_args_1,^activation_quant_18/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�	
IdentityIdentity,dense_noise/StatefulPartitionedCall:output:0,^activation_quant_14/StatefulPartitionedCall<^activation_quant_14/relux/Regularizer/Square/ReadVariableOp,^activation_quant_15/StatefulPartitionedCall<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp,^activation_quant_16/StatefulPartitionedCall<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp,^activation_quant_17/StatefulPartitionedCall<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp,^activation_quant_18/StatefulPartitionedCall<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall(^conv2d_noise_17/StatefulPartitionedCall(^conv2d_noise_18/StatefulPartitionedCall(^conv2d_noise_19/StatefulPartitionedCall(^conv2d_noise_20/StatefulPartitionedCall$^dense_noise/StatefulPartitionedCall(^noise_injection/StatefulPartitionedCall*^noise_injection_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::2Z
+activation_quant_14/StatefulPartitionedCall+activation_quant_14/StatefulPartitionedCall2z
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp;activation_quant_14/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_15/StatefulPartitionedCall+activation_quant_15/StatefulPartitionedCall2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_16/StatefulPartitionedCall+activation_quant_16/StatefulPartitionedCall2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_17/StatefulPartitionedCall+activation_quant_17/StatefulPartitionedCall2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_18/StatefulPartitionedCall+activation_quant_18/StatefulPartitionedCall2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2R
'conv2d_noise_17/StatefulPartitionedCall'conv2d_noise_17/StatefulPartitionedCall2R
'conv2d_noise_18/StatefulPartitionedCall'conv2d_noise_18/StatefulPartitionedCall2R
'conv2d_noise_19/StatefulPartitionedCall'conv2d_noise_19/StatefulPartitionedCall2R
'conv2d_noise_20/StatefulPartitionedCall'conv2d_noise_20/StatefulPartitionedCall2J
#dense_noise/StatefulPartitionedCall#dense_noise/StatefulPartitionedCall2R
'noise_injection/StatefulPartitionedCall'noise_injection/StatefulPartitionedCall2V
)noise_injection_1/StatefulPartitionedCall)noise_injection_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
0__inference_conv2d_noise_19_layer_call_fn_417589
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_4153792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
��
�
A__inference_model_layer_call_and_return_conditional_losses_416005

inputs
inputs_16
2activation_quant_14_statefulpartitionedcall_args_12
.conv2d_noise_17_statefulpartitionedcall_args_12
.conv2d_noise_17_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_19
5batch_normalization_15_statefulpartitionedcall_args_29
5batch_normalization_15_statefulpartitionedcall_args_39
5batch_normalization_15_statefulpartitionedcall_args_46
2activation_quant_15_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_46
2activation_quant_16_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_46
2activation_quant_17_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_19
5batch_normalization_18_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_39
5batch_normalization_18_statefulpartitionedcall_args_46
2activation_quant_18_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_2
identity��+activation_quant_14/StatefulPartitionedCall�;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�+activation_quant_15/StatefulPartitionedCall�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�+activation_quant_16/StatefulPartitionedCall�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�+activation_quant_17/StatefulPartitionedCall�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�+activation_quant_18/StatefulPartitionedCall�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�.batch_normalization_15/StatefulPartitionedCall�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�'conv2d_noise_17/StatefulPartitionedCall�'conv2d_noise_18/StatefulPartitionedCall�'conv2d_noise_19/StatefulPartitionedCall�'conv2d_noise_20/StatefulPartitionedCall�#dense_noise/StatefulPartitionedCall�'noise_injection/StatefulPartitionedCall�)noise_injection_1/StatefulPartitionedCall�
'noise_injection/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_noise_injection_layer_call_and_return_conditional_losses_4148982)
'noise_injection/StatefulPartitionedCall�
)noise_injection_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1(^noise_injection/StatefulPartitionedCall*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_4149262+
)noise_injection_1/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall0noise_injection/StatefulPartitionedCall:output:02noise_injection_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4149492
add/PartitionedCall�
+activation_quant_14/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:02activation_quant_14_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_14_layer_call_and_return_conditional_losses_4149782-
+activation_quant_14/StatefulPartitionedCall�
'conv2d_noise_17/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_14/StatefulPartitionedCall:output:0.conv2d_noise_17_statefulpartitionedcall_args_1.conv2d_noise_17_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_4150182)
'conv2d_noise_17/StatefulPartitionedCall�
.batch_normalization_15/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_17/StatefulPartitionedCall:output:05batch_normalization_15_statefulpartitionedcall_args_15batch_normalization_15_statefulpartitionedcall_args_25batch_normalization_15_statefulpartitionedcall_args_35batch_normalization_15_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_41508020
.batch_normalization_15/StatefulPartitionedCall�
+activation_quant_15/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_15/StatefulPartitionedCall:output:02activation_quant_15_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_4151462-
+activation_quant_15/StatefulPartitionedCall�
'conv2d_noise_18/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_15/StatefulPartitionedCall:output:0.conv2d_noise_18_statefulpartitionedcall_args_1.conv2d_noise_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_4151862)
'conv2d_noise_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_41524820
.batch_normalization_16/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall4activation_quant_14/StatefulPartitionedCall:output:07batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4153002
add_1/PartitionedCall�
+activation_quant_16/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:02activation_quant_16_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_4153292-
+activation_quant_16/StatefulPartitionedCall�
'conv2d_noise_19/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:0.conv2d_noise_19_statefulpartitionedcall_args_1.conv2d_noise_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_4153692)
'conv2d_noise_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_41543120
.batch_normalization_17/StatefulPartitionedCall�
+activation_quant_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:02activation_quant_17_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_4154972-
+activation_quant_17/StatefulPartitionedCall�
'conv2d_noise_20/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_17/StatefulPartitionedCall:output:0.conv2d_noise_20_statefulpartitionedcall_args_1.conv2d_noise_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_4155372)
'conv2d_noise_20/StatefulPartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_20/StatefulPartitionedCall:output:05batch_normalization_18_statefulpartitionedcall_args_15batch_normalization_18_statefulpartitionedcall_args_25batch_normalization_18_statefulpartitionedcall_args_35batch_normalization_18_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_41559920
.batch_normalization_18/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:07batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4156512
add_2/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_4148762#
!average_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_4156672
flatten/PartitionedCall�
+activation_quant_18/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:02activation_quant_18_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_4156952-
+activation_quant_18/StatefulPartitionedCall�
#dense_noise/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_18/StatefulPartitionedCall:output:0*dense_noise_statefulpartitionedcall_args_1*dense_noise_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_4157362%
#dense_noise/StatefulPartitionedCall�
;activation_quant_14/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_14_statefulpartitionedcall_args_1,^activation_quant_14/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_14/relux/Regularizer/SquareSquareCactivation_quant_14/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_14/relux/Regularizer/Square�
+activation_quant_14/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_14/relux/Regularizer/Const�
)activation_quant_14/relux/Regularizer/SumSum0activation_quant_14/relux/Regularizer/Square:y:04activation_quant_14/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/Sum�
+activation_quant_14/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_14/relux/Regularizer/mul/x�
)activation_quant_14/relux/Regularizer/mulMul4activation_quant_14/relux/Regularizer/mul/x:output:02activation_quant_14/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/mul�
+activation_quant_14/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_14/relux/Regularizer/add/x�
)activation_quant_14/relux/Regularizer/addAddV24activation_quant_14/relux/Regularizer/add/x:output:0-activation_quant_14/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_14/relux/Regularizer/add�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_15_statefulpartitionedcall_args_1,^activation_quant_15/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_16_statefulpartitionedcall_args_1,^activation_quant_16/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_17_statefulpartitionedcall_args_1,^activation_quant_17/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_18_statefulpartitionedcall_args_1,^activation_quant_18/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�	
IdentityIdentity,dense_noise/StatefulPartitionedCall:output:0,^activation_quant_14/StatefulPartitionedCall<^activation_quant_14/relux/Regularizer/Square/ReadVariableOp,^activation_quant_15/StatefulPartitionedCall<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp,^activation_quant_16/StatefulPartitionedCall<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp,^activation_quant_17/StatefulPartitionedCall<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp,^activation_quant_18/StatefulPartitionedCall<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp/^batch_normalization_15/StatefulPartitionedCall/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall(^conv2d_noise_17/StatefulPartitionedCall(^conv2d_noise_18/StatefulPartitionedCall(^conv2d_noise_19/StatefulPartitionedCall(^conv2d_noise_20/StatefulPartitionedCall$^dense_noise/StatefulPartitionedCall(^noise_injection/StatefulPartitionedCall*^noise_injection_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@:::::::::::::::::::::::::::::::2Z
+activation_quant_14/StatefulPartitionedCall+activation_quant_14/StatefulPartitionedCall2z
;activation_quant_14/relux/Regularizer/Square/ReadVariableOp;activation_quant_14/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_15/StatefulPartitionedCall+activation_quant_15/StatefulPartitionedCall2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_16/StatefulPartitionedCall+activation_quant_16/StatefulPartitionedCall2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_17/StatefulPartitionedCall+activation_quant_17/StatefulPartitionedCall2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_18/StatefulPartitionedCall+activation_quant_18/StatefulPartitionedCall2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2`
.batch_normalization_15/StatefulPartitionedCall.batch_normalization_15/StatefulPartitionedCall2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2R
'conv2d_noise_17/StatefulPartitionedCall'conv2d_noise_17/StatefulPartitionedCall2R
'conv2d_noise_18/StatefulPartitionedCall'conv2d_noise_18/StatefulPartitionedCall2R
'conv2d_noise_19/StatefulPartitionedCall'conv2d_noise_19/StatefulPartitionedCall2R
'conv2d_noise_20/StatefulPartitionedCall'conv2d_noise_20/StatefulPartitionedCall2J
#dense_noise/StatefulPartitionedCall#dense_noise/StatefulPartitionedCall2R
'noise_injection/StatefulPartitionedCall'noise_injection/StatefulPartitionedCall2V
)noise_injection_1/StatefulPartitionedCall)noise_injection_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������@
C
input_28
serving_default_input_2:0���������@?
dense_noise0
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer-19
layer-20
layer_with_weights-12
layer-21
layer_with_weights-13
layer-22
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�
_tf_keras_model�{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "input_spec": [null, null], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model"}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-06, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 8, 8, 64], "config": {"batch_input_shape": [null, 8, 8, 64], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 8, 8, 64], "config": {"batch_input_shape": [null, 8, 8, 64], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�
	variables
regularization_losses
 trainable_variables
!	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "noise_injection", "name": "noise_injection", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
"	variables
#regularization_losses
$trainable_variables
%	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "noise_injection", "name": "noise_injection_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
&	variables
'regularization_losses
(trainable_variables
)	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add", "trainable": true, "dtype": "float32"}}
�
	*relux
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "activation_quant", "name": "activation_quant_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�

/kernel
0bias
1	variables
2regularization_losses
3trainable_variables
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "conv2d_noise", "name": "conv2d_noise_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_15", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
	>relux
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "activation_quant", "name": "activation_quant_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�

Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "conv2d_noise", "name": "conv2d_noise_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}}
�
	Vrelux
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "activation_quant", "name": "activation_quant_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�

[kernel
\bias
]	variables
^regularization_losses
_trainable_variables
`	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "conv2d_noise", "name": "conv2d_noise_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
	jrelux
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "activation_quant", "name": "activation_quant_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�

okernel
pbias
q	variables
rregularization_losses
strainable_variables
t	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "conv2d_noise", "name": "conv2d_noise_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{regularization_losses
|trainable_variables
}	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
~	variables
regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}}
�
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [8, 8], "padding": "valid", "strides": [8, 8], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

�relux
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "activation_quant", "name": "activation_quant_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�
�kernel
	�bias
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "dense_noise", "name": "dense_noise", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate*m�/m�0m�6m�7m�>m�Cm�Dm�Jm�Km�Vm�[m�\m�bm�cm�jm�om�pm�vm�wm�	�m�	�m�	�m�*v�/v�0v�6v�7v�>v�Cv�Dv�Jv�Kv�Vv�[v�\v�bv�cv�jv�ov�pv�vv�wv�	�v�	�v�	�v�"
	optimizer
�
*0
/1
02
63
74
85
96
>7
C8
D9
J10
K11
L12
M13
V14
[15
\16
b17
c18
d19
e20
j21
o22
p23
v24
w25
x26
y27
�28
�29
�30"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
�
*0
/1
02
63
74
>5
C6
D7
J8
K9
V10
[11
\12
b13
c14
j15
o16
p17
v18
w19
�20
�21
�22"
trackable_list_wrapper
�
�metrics
	variables
�layers
regularization_losses
trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
	variables
�layers
regularization_losses
 trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
"	variables
�layers
#regularization_losses
$trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
&	variables
�layers
'regularization_losses
(trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:! 2activation_quant_14/relux
'
*0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
�
�metrics
+	variables
�layers
,regularization_losses
-trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv2d_noise_17/kernel
": @2conv2d_noise_17/bias
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
�
�metrics
1	variables
�layers
2regularization_losses
3trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_15/gamma
):'@2batch_normalization_15/beta
2:0@ (2"batch_normalization_15/moving_mean
6:4@ (2&batch_normalization_15/moving_variance
<
60
71
82
93"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
�
�metrics
:	variables
�layers
;regularization_losses
<trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:! 2activation_quant_15/relux
'
>0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
'
>0"
trackable_list_wrapper
�
�metrics
?	variables
�layers
@regularization_losses
Atrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv2d_noise_18/kernel
": @2conv2d_noise_18/bias
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
�
�metrics
E	variables
�layers
Fregularization_losses
Gtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_16/gamma
):'@2batch_normalization_16/beta
2:0@ (2"batch_normalization_16/moving_mean
6:4@ (2&batch_normalization_16/moving_variance
<
J0
K1
L2
M3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
�
�metrics
N	variables
�layers
Oregularization_losses
Ptrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
R	variables
�layers
Sregularization_losses
Ttrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:! 2activation_quant_16/relux
'
V0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
'
V0"
trackable_list_wrapper
�
�metrics
W	variables
�layers
Xregularization_losses
Ytrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv2d_noise_19/kernel
": @2conv2d_noise_19/bias
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
�
�metrics
]	variables
�layers
^regularization_losses
_trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_17/gamma
):'@2batch_normalization_17/beta
2:0@ (2"batch_normalization_17/moving_mean
6:4@ (2&batch_normalization_17/moving_variance
<
b0
c1
d2
e3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
�
�metrics
f	variables
�layers
gregularization_losses
htrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:! 2activation_quant_17/relux
'
j0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
'
j0"
trackable_list_wrapper
�
�metrics
k	variables
�layers
lregularization_losses
mtrainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv2d_noise_20/kernel
": @2conv2d_noise_20/bias
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
�
�metrics
q	variables
�layers
rregularization_losses
strainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_18/gamma
):'@2batch_normalization_18/beta
2:0@ (2"batch_normalization_18/moving_mean
6:4@ (2&batch_normalization_18/moving_variance
<
v0
w1
x2
y3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
�
�metrics
z	variables
�layers
{regularization_losses
|trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
~	variables
�layers
regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:! 2activation_quant_18/relux
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
$:"@
2dense_noise/kernel
:
2dense_noise/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
�0"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
 "
trackable_list_wrapper
X
80
91
L2
M3
d4
e5
x6
y7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�
_fn_kwargs
�	variables
�regularization_losses
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�	variables
�layers
�regularization_losses
�trainable_variables
 �layer_regularization_losses
�non_trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(:& 2 Adam/activation_quant_14/relux/m
5:3@@2Adam/conv2d_noise_17/kernel/m
':%@2Adam/conv2d_noise_17/bias/m
/:-@2#Adam/batch_normalization_15/gamma/m
.:,@2"Adam/batch_normalization_15/beta/m
(:& 2 Adam/activation_quant_15/relux/m
5:3@@2Adam/conv2d_noise_18/kernel/m
':%@2Adam/conv2d_noise_18/bias/m
/:-@2#Adam/batch_normalization_16/gamma/m
.:,@2"Adam/batch_normalization_16/beta/m
(:& 2 Adam/activation_quant_16/relux/m
5:3@@2Adam/conv2d_noise_19/kernel/m
':%@2Adam/conv2d_noise_19/bias/m
/:-@2#Adam/batch_normalization_17/gamma/m
.:,@2"Adam/batch_normalization_17/beta/m
(:& 2 Adam/activation_quant_17/relux/m
5:3@@2Adam/conv2d_noise_20/kernel/m
':%@2Adam/conv2d_noise_20/bias/m
/:-@2#Adam/batch_normalization_18/gamma/m
.:,@2"Adam/batch_normalization_18/beta/m
(:& 2 Adam/activation_quant_18/relux/m
):'@
2Adam/dense_noise/kernel/m
#:!
2Adam/dense_noise/bias/m
(:& 2 Adam/activation_quant_14/relux/v
5:3@@2Adam/conv2d_noise_17/kernel/v
':%@2Adam/conv2d_noise_17/bias/v
/:-@2#Adam/batch_normalization_15/gamma/v
.:,@2"Adam/batch_normalization_15/beta/v
(:& 2 Adam/activation_quant_15/relux/v
5:3@@2Adam/conv2d_noise_18/kernel/v
':%@2Adam/conv2d_noise_18/bias/v
/:-@2#Adam/batch_normalization_16/gamma/v
.:,@2"Adam/batch_normalization_16/beta/v
(:& 2 Adam/activation_quant_16/relux/v
5:3@@2Adam/conv2d_noise_19/kernel/v
':%@2Adam/conv2d_noise_19/bias/v
/:-@2#Adam/batch_normalization_17/gamma/v
.:,@2"Adam/batch_normalization_17/beta/v
(:& 2 Adam/activation_quant_17/relux/v
5:3@@2Adam/conv2d_noise_20/kernel/v
':%@2Adam/conv2d_noise_20/bias/v
/:-@2#Adam/batch_normalization_18/gamma/v
.:,@2"Adam/batch_normalization_18/beta/v
(:& 2 Adam/activation_quant_18/relux/v
):'@
2Adam/dense_noise/kernel/v
#:!
2Adam/dense_noise/bias/v
�2�
A__inference_model_layer_call_and_return_conditional_losses_415807
A__inference_model_layer_call_and_return_conditional_losses_416857
A__inference_model_layer_call_and_return_conditional_losses_415904
A__inference_model_layer_call_and_return_conditional_losses_416663�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_414342�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *^�[
Y�V
)�&
input_1���������@
)�&
input_2���������@
�2�
&__inference_model_layer_call_fn_416931
&__inference_model_layer_call_fn_416173
&__inference_model_layer_call_fn_416039
&__inference_model_layer_call_fn_416894�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_noise_injection_layer_call_and_return_conditional_losses_416942
K__inference_noise_injection_layer_call_and_return_conditional_losses_416946�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_noise_injection_layer_call_fn_416951
0__inference_noise_injection_layer_call_fn_416956�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_416971
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_416967�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_noise_injection_1_layer_call_fn_416981
2__inference_noise_injection_1_layer_call_fn_416976�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_add_layer_call_and_return_conditional_losses_416987�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_add_layer_call_fn_416993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_activation_quant_14_layer_call_and_return_conditional_losses_417021�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_activation_quant_14_layer_call_fn_417027�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_417067
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_417057�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_conv2d_noise_17_layer_call_fn_417081
0__inference_conv2d_noise_17_layer_call_fn_417074�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417149
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417201
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417223
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417127�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_15_layer_call_fn_417158
7__inference_batch_normalization_15_layer_call_fn_417241
7__inference_batch_normalization_15_layer_call_fn_417232
7__inference_batch_normalization_15_layer_call_fn_417167�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_417269�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_activation_quant_15_layer_call_fn_417275�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_417305
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_417315�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_conv2d_noise_18_layer_call_fn_417322
0__inference_conv2d_noise_18_layer_call_fn_417329�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417397
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417471
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417449
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417375�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_16_layer_call_fn_417415
7__inference_batch_normalization_16_layer_call_fn_417480
7__inference_batch_normalization_16_layer_call_fn_417489
7__inference_batch_normalization_16_layer_call_fn_417406�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_add_1_layer_call_and_return_conditional_losses_417495�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_add_1_layer_call_fn_417501�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_417529�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_activation_quant_16_layer_call_fn_417535�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_417575
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_417565�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_conv2d_noise_19_layer_call_fn_417582
0__inference_conv2d_noise_19_layer_call_fn_417589�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417709
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417657
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417635
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417731�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_17_layer_call_fn_417740
7__inference_batch_normalization_17_layer_call_fn_417666
7__inference_batch_normalization_17_layer_call_fn_417749
7__inference_batch_normalization_17_layer_call_fn_417675�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_417777�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_activation_quant_17_layer_call_fn_417783�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_417823
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_417813�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_conv2d_noise_20_layer_call_fn_417837
0__inference_conv2d_noise_20_layer_call_fn_417830�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417883
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417905
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417957
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417979�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_18_layer_call_fn_417914
7__inference_batch_normalization_18_layer_call_fn_417988
7__inference_batch_normalization_18_layer_call_fn_417923
7__inference_batch_normalization_18_layer_call_fn_417997�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_add_2_layer_call_and_return_conditional_losses_418003�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_add_2_layer_call_fn_418009�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_414876�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
2__inference_average_pooling2d_layer_call_fn_414882�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
C__inference_flatten_layer_call_and_return_conditional_losses_418015�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_flatten_layer_call_fn_418020�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_418048�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_activation_quant_18_layer_call_fn_418054�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_dense_noise_layer_call_and_return_conditional_losses_418096
G__inference_dense_noise_layer_call_and_return_conditional_losses_418085�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_dense_noise_layer_call_fn_418103
,__inference_dense_noise_layer_call_fn_418110�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference_loss_fn_0_418123�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_418136�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_418149�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_418162�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_4_418175�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
:B8
$__inference_signature_wrapper_416307input_1input_2
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
!__inference__wrapped_model_414342�"*/06789>CDJKLMV[\bcdejopvwxy���h�e
^�[
Y�V
)�&
input_1���������@
)�&
input_2���������@
� "9�6
4
dense_noise%�"
dense_noise���������
�
O__inference_activation_quant_14_layer_call_and_return_conditional_losses_417021f*2�/
(�%
#� 
x���������@
� "-�*
#� 
0���������@
� �
4__inference_activation_quant_14_layer_call_fn_417027Y*2�/
(�%
#� 
x���������@
� " ����������@�
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_417269f>2�/
(�%
#� 
x���������@
� "-�*
#� 
0���������@
� �
4__inference_activation_quant_15_layer_call_fn_417275Y>2�/
(�%
#� 
x���������@
� " ����������@�
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_417529fV2�/
(�%
#� 
x���������@
� "-�*
#� 
0���������@
� �
4__inference_activation_quant_16_layer_call_fn_417535YV2�/
(�%
#� 
x���������@
� " ����������@�
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_417777fj2�/
(�%
#� 
x���������@
� "-�*
#� 
0���������@
� �
4__inference_activation_quant_17_layer_call_fn_417783Yj2�/
(�%
#� 
x���������@
� " ����������@�
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_418048W�*�'
 �
�
x���������@
� "%�"
�
0���������@
� �
4__inference_activation_quant_18_layer_call_fn_418054J�*�'
 �
�
x���������@
� "����������@�
A__inference_add_1_layer_call_and_return_conditional_losses_417495�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� "-�*
#� 
0���������@
� �
&__inference_add_1_layer_call_fn_417501�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� " ����������@�
A__inference_add_2_layer_call_and_return_conditional_losses_418003�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� "-�*
#� 
0���������@
� �
&__inference_add_2_layer_call_fn_418009�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� " ����������@�
?__inference_add_layer_call_and_return_conditional_losses_416987�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� "-�*
#� 
0���������@
� �
$__inference_add_layer_call_fn_416993�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� " ����������@�
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_414876�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_average_pooling2d_layer_call_fn_414882�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417127�6789M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417149�6789M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417201r6789;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_15_layer_call_and_return_conditional_losses_417223r6789;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
7__inference_batch_normalization_15_layer_call_fn_417158�6789M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_15_layer_call_fn_417167�6789M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_15_layer_call_fn_417232e6789;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_15_layer_call_fn_417241e6789;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417375rJKLM;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417397rJKLM;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417449�JKLMM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_417471�JKLMM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
7__inference_batch_normalization_16_layer_call_fn_417406eJKLM;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_16_layer_call_fn_417415eJKLM;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
7__inference_batch_normalization_16_layer_call_fn_417480�JKLMM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_16_layer_call_fn_417489�JKLMM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417635�bcdeM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417657�bcdeM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417709rbcde;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_417731rbcde;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
7__inference_batch_normalization_17_layer_call_fn_417666�bcdeM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_17_layer_call_fn_417675�bcdeM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_17_layer_call_fn_417740ebcde;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_17_layer_call_fn_417749ebcde;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417883�vwxyM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417905�vwxyM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417957rvwxy;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_417979rvwxy;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
7__inference_batch_normalization_18_layer_call_fn_417914�vwxyM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_18_layer_call_fn_417923�vwxyM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_18_layer_call_fn_417988evwxy;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_18_layer_call_fn_417997evwxy;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_417057k/06�3
,�)
#� 
x���������@
p
� "-�*
#� 
0���������@
� �
K__inference_conv2d_noise_17_layer_call_and_return_conditional_losses_417067k/06�3
,�)
#� 
x���������@
p 
� "-�*
#� 
0���������@
� �
0__inference_conv2d_noise_17_layer_call_fn_417074^/06�3
,�)
#� 
x���������@
p
� " ����������@�
0__inference_conv2d_noise_17_layer_call_fn_417081^/06�3
,�)
#� 
x���������@
p 
� " ����������@�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_417305kCD6�3
,�)
#� 
x���������@
p
� "-�*
#� 
0���������@
� �
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_417315kCD6�3
,�)
#� 
x���������@
p 
� "-�*
#� 
0���������@
� �
0__inference_conv2d_noise_18_layer_call_fn_417322^CD6�3
,�)
#� 
x���������@
p
� " ����������@�
0__inference_conv2d_noise_18_layer_call_fn_417329^CD6�3
,�)
#� 
x���������@
p 
� " ����������@�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_417565k[\6�3
,�)
#� 
x���������@
p
� "-�*
#� 
0���������@
� �
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_417575k[\6�3
,�)
#� 
x���������@
p 
� "-�*
#� 
0���������@
� �
0__inference_conv2d_noise_19_layer_call_fn_417582^[\6�3
,�)
#� 
x���������@
p
� " ����������@�
0__inference_conv2d_noise_19_layer_call_fn_417589^[\6�3
,�)
#� 
x���������@
p 
� " ����������@�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_417813kop6�3
,�)
#� 
x���������@
p
� "-�*
#� 
0���������@
� �
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_417823kop6�3
,�)
#� 
x���������@
p 
� "-�*
#� 
0���������@
� �
0__inference_conv2d_noise_20_layer_call_fn_417830^op6�3
,�)
#� 
x���������@
p
� " ����������@�
0__inference_conv2d_noise_20_layer_call_fn_417837^op6�3
,�)
#� 
x���������@
p 
� " ����������@�
G__inference_dense_noise_layer_call_and_return_conditional_losses_418085]��.�+
$�!
�
x���������@
p
� "%�"
�
0���������

� �
G__inference_dense_noise_layer_call_and_return_conditional_losses_418096]��.�+
$�!
�
x���������@
p 
� "%�"
�
0���������

� �
,__inference_dense_noise_layer_call_fn_418103P��.�+
$�!
�
x���������@
p
� "����������
�
,__inference_dense_noise_layer_call_fn_418110P��.�+
$�!
�
x���������@
p 
� "����������
�
C__inference_flatten_layer_call_and_return_conditional_losses_418015`7�4
-�*
(�%
inputs���������@
� "%�"
�
0���������@
� 
(__inference_flatten_layer_call_fn_418020S7�4
-�*
(�%
inputs���������@
� "����������@;
__inference_loss_fn_0_418123*�

� 
� "� ;
__inference_loss_fn_1_418136>�

� 
� "� ;
__inference_loss_fn_2_418149V�

� 
� "� ;
__inference_loss_fn_3_418162j�

� 
� "� <
__inference_loss_fn_4_418175��

� 
� "� �
A__inference_model_layer_call_and_return_conditional_losses_415807�"*/06789>CDJKLMV[\bcdejopvwxy���p�m
f�c
Y�V
)�&
input_1���������@
)�&
input_2���������@
p

 
� "%�"
�
0���������

� �
A__inference_model_layer_call_and_return_conditional_losses_415904�"*/06789>CDJKLMV[\bcdejopvwxy���p�m
f�c
Y�V
)�&
input_1���������@
)�&
input_2���������@
p 

 
� "%�"
�
0���������

� �
A__inference_model_layer_call_and_return_conditional_losses_416663�"*/06789>CDJKLMV[\bcdejopvwxy���r�o
h�e
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
p

 
� "%�"
�
0���������

� �
A__inference_model_layer_call_and_return_conditional_losses_416857�"*/06789>CDJKLMV[\bcdejopvwxy���r�o
h�e
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
p 

 
� "%�"
�
0���������

� �
&__inference_model_layer_call_fn_416039�"*/06789>CDJKLMV[\bcdejopvwxy���p�m
f�c
Y�V
)�&
input_1���������@
)�&
input_2���������@
p

 
� "����������
�
&__inference_model_layer_call_fn_416173�"*/06789>CDJKLMV[\bcdejopvwxy���p�m
f�c
Y�V
)�&
input_1���������@
)�&
input_2���������@
p 

 
� "����������
�
&__inference_model_layer_call_fn_416894�"*/06789>CDJKLMV[\bcdejopvwxy���r�o
h�e
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
p

 
� "����������
�
&__inference_model_layer_call_fn_416931�"*/06789>CDJKLMV[\bcdejopvwxy���r�o
h�e
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
p 

 
� "����������
�
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_416967g6�3
,�)
#� 
x���������@
p
� "-�*
#� 
0���������@
� �
M__inference_noise_injection_1_layer_call_and_return_conditional_losses_416971g6�3
,�)
#� 
x���������@
p 
� "-�*
#� 
0���������@
� �
2__inference_noise_injection_1_layer_call_fn_416976Z6�3
,�)
#� 
x���������@
p
� " ����������@�
2__inference_noise_injection_1_layer_call_fn_416981Z6�3
,�)
#� 
x���������@
p 
� " ����������@�
K__inference_noise_injection_layer_call_and_return_conditional_losses_416942g6�3
,�)
#� 
x���������@
p
� "-�*
#� 
0���������@
� �
K__inference_noise_injection_layer_call_and_return_conditional_losses_416946g6�3
,�)
#� 
x���������@
p 
� "-�*
#� 
0���������@
� �
0__inference_noise_injection_layer_call_fn_416951Z6�3
,�)
#� 
x���������@
p
� " ����������@�
0__inference_noise_injection_layer_call_fn_416956Z6�3
,�)
#� 
x���������@
p 
� " ����������@�
$__inference_signature_wrapper_416307�"*/06789>CDJKLMV[\bcdejopvwxy���y�v
� 
o�l
4
input_1)�&
input_1���������@
4
input_2)�&
input_2���������@"9�6
4
dense_noise%�"
dense_noise���������
