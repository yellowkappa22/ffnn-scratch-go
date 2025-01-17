package main

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"reflect"

	"gonum.org/v1/gonum/mat"
)

//// Data Preparation ////

// Image Converters
func SaveIDX3ToPNG() {

	// Read in the raw image
	file, err := os.Open("mnist/t10k-images.idx3-ubyte")
	if err != nil {
		fmt.Println("Open Error:", err)
		return
	}
	defer file.Close()

	// Decode the raw image and save the first image's pixels
	type IDX3Header struct {
		MagicID uint32
		NImages uint32
		NRows   uint32
		NCols   uint32
	}
	var header IDX3Header

	if err := binary.Read(file, binary.BigEndian, &header); err != nil {
		fmt.Println("Decode Error:", err)
		return
	}

	file_size := header.NRows * header.NCols
	pixels := make([]byte, file_size)

	if _, err := file.Read(pixels); err != nil {
		fmt.Println("Decode Error:", err)
		return
	}

	// Create go image
	img_p := image.NewGray(image.Rect(0, 0, int(header.NRows), int(header.NCols)))
	for y := 0; y < int(header.NRows); y++ {
		for x := 0; x < int(header.NCols); x++ {
			img_p.SetGray(x, y, color.Gray{Y: pixels[y*int(header.NCols)+x]})
		}
	}

	// Save the image as PNG
	outfile, err := os.Create("mystery_letter.png")
	if err != nil {
		fmt.Println("Create Error:", err)
		return
	}
	defer outfile.Close()

	err = png.Encode(outfile, img_p)
	if err != nil {
		fmt.Println("Encode Error:", err)
		return
	}
	fmt.Println("Image saved succesfuly")
}

func ConvertIDX3toMatrixArray(path string) ([]*mat.Dense, error) {
	/*
		Converts Files of Type IDX3-Ubyte to an Array of Matrices
	*/

	// Open File
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file at %s: %w", path, err)
	}
	defer file.Close()

	// Construct Header
	type IDX3Header struct {
		MagicNumber uint32
		NImages     uint32
		NRows       uint32
		NCols       uint32
	}
	var header IDX3Header
	if err = binary.Read(file, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("failed to read binary header: %w", err)
	}

	// Construct Aggregate Pixel Array
	file_size := int(header.NCols * header.NRows)
	pixels := make([]byte, uint32(file_size)*header.NImages)
	if _, err = file.Read(pixels); err != nil {
		return nil, fmt.Errorf("failed to read binary pixels: %w", err)
	}

	// Construct the Structured Pixel Array of Matrices
	feature_array := make([]*mat.Dense, header.NImages)
	for i := 0; i < int(header.NImages); i++ {
		A := mat.NewDense(file_size, 1, nil)
		for j := 0; j < file_size; j++ {
			A.Set(j, 0, float64(pixels[i*file_size+j]))
		}
		feature_array[i] = A
	}

	//
	return feature_array, nil
}

func ConvertIDX1toMatrixArray(path string) ([]*mat.Dense, error) {
	/*
		Converts Files of Type IDX3-Ubyte to an Array of Matrices
	*/

	// Open File
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file at %s: %w", path, err)
	}
	defer file.Close()

	// Construct Header
	type IDX1Header struct {
		MagicNumber uint32
		NLabels     uint32
	}
	var header IDX1Header
	if err = binary.Read(file, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("failed to read binary header: %w", err)
	}

	// Construct Aggregate Pixel Array
	labels := make([]byte, header.NLabels)
	if _, err = file.Read(labels); err != nil {
		return nil, fmt.Errorf("failed to read binary pixels: %w", err)
	}

	// Construct the Structured Pixel Array of Matrices
	target_array := make([]*mat.Dense, header.NLabels)
	for i := 0; i < int(header.NLabels); i++ {
		A := mat.NewDense(10, 1, nil)
		for j := 0; j < 10; j++ {
			if int(labels[i]) == j {
				A.Set(j, 0, 1)
			} else {
				A.Set(j, 0, 0)
			}
		}
		target_array[i] = A
	}

	//
	return target_array, nil
}

////

//// Neural Network ////

// Loss Function
func CrossEntropyLoss(y, y_hat *mat.Dense) float64 {
	// Init Vectors
	dim, _ := y.Dims()

	dimhat, _ := y_hat.Dims()
	fmt.Println("Dim y:", dim)
	fmt.Println("Dim yhat:", dimhat)

	y_hat_log := mat.NewDense(dim, 1, nil)
	y_hat_log.Apply(func(i, j int, v float64) float64 {
		return math.Log(v)
	}, y_hat)
	fmt.Println("Apply went through success")
	S := mat.NewDense(1, 1, nil)
	S.Mul(y.T(), y_hat_log)
	fmt.Println("We don't reach this stage")
	fmt.Println(S)
	fmt.Println(reflect.TypeOf(S))
	loss := -S.At(0, 0)
	fmt.Println("loss success")
	return loss
}

// Activation Functions
func reLU(i, j int, v float64) float64 {
	if v < 0 {
		return 0
	}
	return v
}

func SoftMax(z *mat.Dense) *mat.Dense {
	n_rows, _ := z.Dims()
	a_mat := mat.NewDense(n_rows, 1, nil)
	var sum float64 = 0

	for i := 0; i < n_rows; i++ {
		sum += math.Exp(z.At(i, 0))
	}

	for i := 0; i < n_rows; i++ {
		a := math.Exp(z.At(i, 0)) / sum
		a_mat.Set(i, 0, a)
	}

	return a_mat
}

// Type Structure
type HiddenLayer struct {
	Weights *mat.Dense
	Biases  *mat.Dense
}

func (hl HiddenLayer) ForwardProp(a0 *mat.Dense) (z, a *mat.Dense) {
	m, _ := hl.Weights.Dims()
	_, n := a0.Dims()
	Wa := mat.NewDense(m, n, nil)
	Wa.Mul(hl.Weights, a0)
	z = mat.NewDense(m, n, nil)
	a = mat.NewDense(m, n, nil)
	z.Add(hl.Biases, Wa)
	a.Apply(reLU, z)
	return z, a
}

type OutputLayer struct {
	Weights *mat.Dense
	Biases  *mat.Dense
}

func (ol OutputLayer) ForwardProp(a0 *mat.Dense) (z, a *mat.Dense) {
	m, _ := ol.Weights.Dims()
	_, n := a0.Dims()
	Wa := mat.NewDense(m, n, nil)
	z = mat.NewDense(m, n, nil)
	Wa.Mul(ol.Weights, a0)
	z.Add(ol.Biases, Wa)
	a = SoftMax(z)
	return z, a
}

type LayerInterface interface {
	ForwardProp(*mat.Dense) (z, a *mat.Dense)
}

type NNCache struct {
	Z []*mat.Dense
	A []*mat.Dense
}

type NeuralNetwork struct {
	NInputs    int
	NOutputs   int
	NNeurons   []int
	WeightData []*mat.Dense
	BiasData   []*mat.Dense
	Eta        float64
}

// Method Structure
func (nn *NeuralNetwork) GradientUpdate(c NNCache, y *mat.Dense) {

	/*
		Backwards Propogation
			- SGD Method
			- Featuring "Horrible Linear Algebra Syntax"
			- Delta Rerepresents Cummulative Gradient Product (Aquired Through the Chain Rule)
	*/

	// Initialize Delta
	output_rows, _ := y.Dims()
	aL := c.A[len(c.A)-1]
	delta := mat.NewDense(output_rows, 1, nil)
	delta.Sub(aL, y)

	// Update Gradients
	// L is Last Activation & L-1 is Last Weight
	L := len(c.A) - 1

	for l := L; l > 0; l-- {
		// Weights
		m, k := delta.Dims()
		n, _ := c.A[l-1].Dims()

		P := mat.NewDense(m, n, nil)
		P.Mul(delta, c.A[l-1].T())
		SP := mat.NewDense(m, n, nil)
		SP.Scale(nn.Eta, P)
		W := mat.NewDense(m, n, nil)
		W.Sub(nn.WeightData[l-1], SP)

		nn.WeightData[l-1] = W

		// Biases
		S := mat.NewDense(m, k, nil)
		S.Scale(nn.Eta, delta)
		B := mat.NewDense(m, k, nil)
		B.Sub(nn.BiasData[l-1], S)
		nn.BiasData[l-1] = B

		// Update Delta if Not Last Layer
		if l > 1 {
			Dummy := mat.NewDense(nn.NNeurons[l-2], 1, nil)
			Dummy.Apply(func(i, j int, v float64) float64 {
				if v <= 0 {
					return 0
				} else {
					return 1
				}
			}, c.Z[l-1])

			P = mat.NewDense(n, k, nil) // Question whether both rows and columns should be m (rows of previous delta)
			P.Mul(nn.WeightData[l-1].T(), delta)
			H := mat.NewDense(n, k, nil)
			H.MulElem(P, Dummy)
			delta = H
		}
	}
}

func (c *NNCache) CacheUpdate(nn NeuralNetwork, input *mat.Dense) {

	/*
		Forward Propagation
			- Relying on Custom Defined Layer Interface
				- HiddenLayer Struct
				- OutputLayer Struct
			- Forwardprop()
	*/

	// Initialize Layer Placeholder Variable
	var layer LayerInterface

	// Input Layer
	c.Z[0] = nil
	c.A[0] = input

	// Hidden and Output Layers
	for i := 0; i < len(nn.NNeurons); i++ {
		layer = HiddenLayer{
			Weights: nn.WeightData[i],
			Biases:  nn.BiasData[i],
		}
		c.Z[i+1], c.A[i+1] = layer.ForwardProp(c.A[i])
	}
}

func (nn *NeuralNetwork) Train(trainX, trainY []*mat.Dense, c NNCache) {
	// Dimension Variable Initialization
	train_size := len(trainX)

	for i := 0; i < train_size; i++ {
		// Define Current Target (1 Vector Since SGD)
		y := trainY[i]
		x := trainX[i]

		// Forward Propagation
		c.CacheUpdate(*nn, x)
		// Backward Propagation
		nn.GradientUpdate(c, y)
	}
}

func (nn NeuralNetwork) Predict(input *mat.Dense) (*mat.Dense, int) {
	/*
		Forward Propagation
			- Relying on Custom Defined Layer Interface
				- HiddenLayer Struct
				- OutputLayer Struct
			- Forwardprop()
	*/

	// Initialize Layer Placeholder and Prediction Variable
	var layer LayerInterface
	var y_hat_mat *mat.Dense
	var y_hat_int int

	a := input

	// Hidden and Output Layers
	for i := 0; i < len(nn.NNeurons); i++ {
		layer = HiddenLayer{
			Weights: nn.WeightData[i],
			Biases:  nn.BiasData[i],
		}
		_, a = layer.ForwardProp(a)
	}

	max := float64(0)
	max_idx := -1
	for i := 0; i < 10; i++ {
		if a.At(i, 0) > max {
			max = a.At(i, 0)
			max_idx = i
		}
	}

	y_hat_mat = a
	y_hat_int = max_idx

	return y_hat_mat, y_hat_int
}

// Generate New Neural Network Function
func NewNN(n_inputs int, n_neurons []int, eta float64) (NeuralNetwork, NNCache) {

	/*
		Initializes a neural network struct
			- Metadata: #inputs, #outputs and #neurons
			- Weights and Biases: Random, standard normal, initialization
			- Cache: Empty
			- n_neurons provides both number of layers and dimensions of individual layers
			- Cache items are +1 larger than number of layers to account for inputs
	*/

	// Defining Variables
	n_layers := len(n_neurons)
	weight_data := make([]*mat.Dense, n_layers)
	bias_data := make([]*mat.Dense, n_layers)
	z := make([]*mat.Dense, (1 + n_layers))
	a := make([]*mat.Dense, (1 + n_layers))

	// Populating Variables
	dim := n_inputs
	for i := range n_neurons {
		n := n_neurons[i]

		// Cache
		z_mat := mat.NewDense(n, 1, nil)
		a_mat := mat.NewDense(n, 1, nil)

		// Random Initialization of Weight and Bias Matrices
		w_mat := mat.NewDense(n, dim, nil)
		b_mat := mat.NewDense(n, 1, nil)
		for r := 0; r < n; r++ {
			b_mat.Set(r, 0, rand.NormFloat64())
			for c := 0; c < dim; c++ {
				w_mat.Set(r, c, rand.NormFloat64())
			}
		}
		weight_data[i] = w_mat
		bias_data[i] = b_mat
		z[i] = z_mat
		a[i] = a_mat

		// Update Vector Dimension to Previous Neuron Dimension
		dim = n_neurons[i]
	}
	// Number of outputs & Final Cache Slot
	n_outputs := n_neurons[n_layers-1]
	z_out := mat.NewDense(n_outputs, 1, nil)
	a_out := mat.NewDense(n_outputs, 1, nil)
	z[n_layers] = z_out
	a[n_layers] = a_out
	cache := NNCache{
		Z: z,
		A: a,
	}

	// Construct the Struct
	nn := NeuralNetwork{
		NInputs:    n_inputs,
		NOutputs:   n_outputs,
		NNeurons:   n_neurons,
		WeightData: weight_data,
		BiasData:   bias_data,
		Eta:        eta,
	}

	return nn, cache
}

func ComputeTotalCrossEntropy(nn NeuralNetwork, x, y []*mat.Dense) float64 {
	// Initialize Variables
	var cumm_loss float64
	var mean_loss float64
	var loss float64

	n_obs := len(x)

	// Iteratively Calculate Cummulative Loss Using .Predict() and CrossEntropyLoss()
	for i := 0; i < n_obs; i++ {
		y_hat, _ := nn.Predict(x[i])
		fmt.Println("Initiated y_hat with success")
		loss = CrossEntropyLoss(y[i], y_hat)
		cumm_loss += loss
	}

	mean_loss = -1 / float64(n_obs) * cumm_loss
	return mean_loss
}

////

func main() {
	// Get Feature Array
	train_features, err := ConvertIDX3toMatrixArray("mnist/train-images.idx3-ubyte")
	if err != nil {
		fmt.Println("Error occurred: ", err)
		return
	}

	train_labels, err := ConvertIDX1toMatrixArray("mnist/train-labels.idx1-ubyte")
	if err != nil {
		fmt.Println("Label Error occurred: ", err)
		return
	}

	fmt.Println("Train File Conversion Success")

	test_features, err := ConvertIDX3toMatrixArray("mnist/t10k-images.idx3-ubyte")
	if err != nil {
		fmt.Println("Error occurred: ", err)
		return
	}

	test_labels, err := ConvertIDX1toMatrixArray("mnist/t10k-labels.idx1-ubyte")
	if err != nil {
		fmt.Println("Label Error occurred: ", err)
		return
	}

	fmt.Println("Validation File Conversion Success")
	//

	// Run Neural Network
	n_inputs, _ := train_features[0].Dims()
	n_outputs := 10
	n_neurons := []int{
		300,
		200,
		100,
		n_outputs,
	}
	eta := 0.01

	var NN NeuralNetwork
	var C NNCache

	NN, C = NewNN(n_inputs, n_neurons, eta)
	fmt.Println("Init Success")
	fmt.Println("Dimensions of test:", len(test_features), len(test_labels))
	loss1 := ComputeTotalCrossEntropy(NN, test_features, test_labels)
	fmt.Println("Loss Before Training:", loss1)

	NN.Train(train_features, train_labels, C)
	loss2 := ComputeTotalCrossEntropy(NN, test_features, test_labels)
	fmt.Println("Loss After Training:", loss2)
	fmt.Println("Loss Before Training:", loss1)

	//
}
