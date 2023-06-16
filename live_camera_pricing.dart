import 'dart:async';
import 'dart:typed_data';
import 'dart:isolate';
import 'dart:io';
import 'package:image/image.dart' as image_utils;
import 'package:intl/intl.dart';
import 'package:shufflebuy/experiments/machine_learning/tensorflow/data_prices/data_prices.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:developer';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';


class TensorCameraHomePage extends StatefulWidget {
  final CameraDescription camera;
  const TensorCameraHomePage({Key? key, required this.camera}) : super(key: key);
  @override
  _TensorCameraHomePageState createState() => _TensorCameraHomePageState();
}

class _TensorCameraHomePageState extends State<TensorCameraHomePage> {

  static const modelPath = 'assets/mobilenet/mobilenet_v1_1.0_224_quant.tflite';
  static const labelsPath = 'assets/mobilenet/labels.txt';

  late final Interpreter interpreter;
  late final List<String> labels;

  Tensor? inputTensor;
  Tensor? outputTensor;

  final imagePicker = ImagePicker();
  String? imagePath;
  img.Image? image;


// Add a new Map to hold the selected state for each item
  Map<String, bool> selectionStatus = {};
  List<AverageGeneratedItemData> item_matched_data_objects = [];

  // Add a String to hold the sentence
  String sentence = '';
  String priceDisplay = '';

  //----------------------------------------------------------------------------------------------------
  late CameraController controller;
  img.Image? latestImage;

  @override
  void initState() {
    super.initState();
    // Initialize camera
    controller = CameraController(widget.camera, ResolutionPreset.medium, imageFormatGroup: ImageFormatGroup.jpeg);     //set the image format of the controller YUV_420_888
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      // Start image stream here
      controller.startImageStream((image) => processCameraImage(image));
      setState(() {});
    });
    // Load model and labels from assets
    loadModel();
    loadLabels();
  }


  void processCameraImage(CameraImage? cameraImage) async {
    // Check if cameraImage is not null
    if (cameraImage == null) {
      print('cameraImage is null');
      return;
    }

    // Convert YUV image to RGB
    final img.Image convertedImage = _convertYUV420toImageColor(cameraImage);
    image = convertedImage;

    // Assign the processed image to latestImage
    latestImage = image;

    // Process image
    await processImage();
    setState(() {});
  }
  img.Image _convertYUV420toImageColor(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;

    var imgLib = img.Image(width, height);  // Create imgLib Image buffer

    // Fill imgLib image buffer with R, G, B values. This operation could be optimized
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = uvPixelStride * (x/2).floor() + uvRowStride*(y/2).floor();
        final int index = y * width + x;

        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];

        // Calculate pixel color
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91).round().clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

        // Set pixel color
        imgLib.data[index] = (0xFF<<24) | (b<<16) | (g<<8) | r;
      }
    }

    return imgLib;
  }


  //override setState to avoid calling setState if the widget is not mounted
  @override
  void setState(fn) {
    if (mounted) {
      super.setState(fn);
    }
  }

  // dispose camera controller
  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }
  // Assume boundingBoxes is a list of maps, where each map contains the details of each bounding box
  // like {'label': 'dog', 'x': 100, 'y': 200, 'width': 150, 'height': 150}
  List<Map<String, dynamic>> boundingBoxes = [];
  @override
  Widget build(BuildContext context) {
    if (!controller.value.isInitialized) {
      return Container();
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Object Detection'),
        backgroundColor: Colors.indigo,
      ),
      body: Stack(
        children: <Widget>[
          Positioned.fill(
            child: latestImage == null || boundingBoxes.isEmpty
                ? CameraPreview(controller)
                : FittedBox(
              fit: BoxFit.cover,
              child: Image.memory(Uint8List.fromList(img.encodeJpg(latestImage!))),
            ),
          ),
          ...boundingBoxes.map((box) {
            return Positioned(
              left: box['x'].toDouble(),
              top: box['y'].toDouble(),
              child: Container(
                width: box['width'].toDouble(),
                height: box['height'].toDouble(),
                decoration: BoxDecoration(
                  border: Border.all(
                    color: Colors.red, // Updated color to bright red
                    width: 2,
                  ),
                ),
                child: Text(
                  box['label'],
                  style: TextStyle(
                    color: Colors.red, // Updated color to bright red
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            );
          }).toList(),
        ],
      ),
    );
  }


  // Clean old results when press some take picture button
  void cleanResult() {
    imagePath = null;
    image = null;
    // classification = null;
    setState(() {});
  }

  // Load model
  Future<void> loadModel() async {
    final options = InterpreterOptions();

    // Use XNNPACK Delegate
    if (Platform.isAndroid) {
      options.addDelegate(XNNPackDelegate());
    }

    // Use GPU Delegate
    // doesn't work on emulator
    // if (Platform.isAndroid) {
    //   options.addDelegate(GpuDelegateV2());
    // }

    // Use Metal Delegate
    if (Platform.isIOS) {
      options.addDelegate(GpuDelegate());
    }

    // Load model from assets
    interpreter = await Interpreter.fromAsset(modelPath, options: options);
    // Get tensor input shape [1, 224, 224, 3]
    inputTensor = interpreter.getInputTensors().first;
    // Get tensor output shape [1, 1001]
    outputTensor = interpreter.getOutputTensors().first;
    setState(() {});

    log('Interpreter loaded successfully');
  }

  // Load labels from assets
  Future<void> loadLabels() async {
    final labelTxt = await rootBundle.loadString(labelsPath);
    labels = labelTxt.split('\n');
  }

  // Process picked image
  Future<void> processImage() async {
    // Resize the image to the input size of the model
    img.Image resizedImage = img.copyResize(image!, width: 224, height: 224);

    // Convert the resized image to a byte list
    List<int> byteList = resizedImage.getBytes();

    // Preprocess the image bytes if necessary
    // ...

    // Resize image for model input (Mobilenet use [224, 224])
    final imageInput = img.copyResize(
      image!,
      width: 224,
      height: 224,
    );

    // Get image matrix representation [224, 224, 3]
    final imageMatrix = List.generate(
      imageInput.height,
          (y) => List.generate(
        imageInput.width,
            (x) {
          final pixel = imageInput.getPixel(x, y);
          return [pixel >> 16 & 0xFF, pixel >> 8 & 0xFF, pixel & 0xFF];
        },
      ),
    );

    // Run model inference
    debugPrint('Running inference on image $imagePath');
    List<double> modelOutput = await runModelOnFrame(imageMatrix);

    // Extract the bounding boxes from the output
    boundingBoxes = extractBoundingBoxes(modelOutput);

    // Trigger a rebuild
    setState(() {});
  }

  Future<List<double>> runModelOnFrame(List<List<List<int>>> imageMatrix) async {
    // Preprocessing to convert imageMatrix to the appropriate input format goes here...

    // Get the height and width of the image
    int height = imageMatrix.length;
    int width = imageMatrix[0].length;

    // Prepare the input buffer
    var inputBuffer = List.generate(1, (_) => List.generate(height, (_) => List.generate(width, (_) => List.filled(3, 0.0))));

    // Write the image data into the inputBuffer
    for (var i = 0; i < height; i++) {
      for (var j = 0; j < width; j++) {
        for (var k = 0; k < 3; k++) {
          inputBuffer[0][i][j][k] = imageMatrix[i][j][k] / 255.0; // Normalize the pixel values
        }
      }
    }

    // Prepare the output buffer
    var outputBuffer = List.generate(1, (_) => List<double>.filled(1001, 0.0));

    // Run the model
    interpreter.run(inputBuffer, outputBuffer);

    //print out if the model is running
    debugPrint('Model is running');

    // The outputBuffer now contains the output of the model
    // Flatten the outputBuffer to match the return type of the function
    return outputBuffer.expand((x) => x).toList();
  }




  List<Map<String, dynamic>> extractBoundingBoxes(List<dynamic> outputList) {
    debugPrint('extractBoundingBoxes Output list: $outputList');
    // Assume outputList contains the bounding box coordinates
    List<Map<String, dynamic>> boxes = [];

    for (var boxTensor in outputList) {
      Float32List box = boxTensor.buffer.asFloat32List(); // convert Tensor to Float32List
      double ymin = box[0];
      double xmin = box[1];
      double ymax = box[2];
      double xmax = box[3];

      // Convert the relative coordinates to absolute coordinates
      int imgHeight = image!.height;
      int imgWidth = image!.width;

      int left = (xmin * imgWidth).toInt();
      int top = (ymin * imgHeight).toInt();
      int width = ((xmax - xmin) * imgWidth).toInt();
      int height = ((ymax - ymin) * imgHeight).toInt();

      boxes.add({
        'label': 'Object',
        'x': left,
        'y': top,
        'width': width,
        'height': height,
      });
    }

    return boxes;
  }



  // Run inference
  Future<void> runInference(
      List<List<List<num>>> imageMatrix,
      ) async {
    // Set tensor input [1, 224, 224, 3]
    final input = [imageMatrix];
    // Set tensor output [1, 1001]
    final output = [List<int>.filled(1001, 0)];



    // Run inference
    interpreter.run(input, output);

    // Get first output tensor
    final result = output.first;

    // Set classification map {label: points}
    Map<String, int>? classification = <String, int>{};

    for (var i = 0; i < result.length; i++) {
      if (result[i] != 0) {
        // Set label: points
        classification![labels[i]] = result[i];
        //Generate item data model from matching pricing_data_array
        //lookup price:
        double price_result = 0.0;
        pricing_data_array.forEach((key, value) {
          if (key.toString().toLowerCase() == labels[i].toString().toLowerCase()) {
            price_result = value;
          }
        });
        AverageGeneratedItemData itemData = AverageGeneratedItemData(name: labels[i], price: price_result, ranking: result[i]);
        item_matched_data_objects.add(itemData);
      }
    }

    setState(() {});
  }

  Map pricing_data_array = {
    "background": 0,
    "tench": 10.0,
    "goldfish": 5.0,
    "great white shark": 10000.0,
    "tiger shark": 8000.0,
    "hammerhead": 6000.0,
    "electric ray": 20.0,
    "stingray": 30.0,
    "cock": 5.0,
    "hen": 5.0,
    "ostrich": 500.0,
    "brambling": 15.0,
    "goldfinch": 10.0,
    "house finch": 8.0,
    "junco": 7.0,
    "indigo bunting": 20.0,
    "robin": 10.0,
    "bulbul": 25.0,
    "jay": 15.0,
    "magpie": 20.0,
    "chickadee": 10.0,
    "water ouzel": 30.0,
    "kite": 50.0,
    "bald eagle": 1000.0,
    "vulture": 200.0,
    "great grey owl": 500.0,
    "European fire salamander": 50.0,
    "common newt": 10.0,
    "eft": 10.0,
    "spotted salamander": 20.0,
    "axolotl": 30.0,
    "bullfrog": 15.0,
    "tree frog": 10.0,
    "tailed frog": 20.0,
    "loggerhead": 500.0,
    "leatherback turtle": 1000.0,
    "mud turtle": 20.0,
    "terrapin": 30.0,
    "box turtle": 50.0,
    "banded gecko": 40.0,
    "common iguana": 200.0,
    "American chameleon": 150.0,
    "whiptail": 30.0,
    "agama": 25.0,
    "frilled lizard": 100.0,
    "alligator lizard": 50.0,
    "Gila monster": 300.0,
    "green lizard": 20.0,
    "African chameleon": 150.0,
    "Komodo dragon": 5000.0,
    "African crocodile": 2000.0,
    "American alligator": 1500.0,
    "triceratops": 1000000.0,
    "thunder snake": 20.0,
    "ringneck snake": 10.0,
    "hognose snake": 30.0,
    "green snake": 10.0,
    "king snake": 50.0,
    "garter snake": 15.0,
    "water snake": 20.0,
    "vine snake": 25.0,
    "night snake": 15.0,
    "boa constrictor": 200.0,
    "rock python": 300.0,
    "Indian cobra": 100.0,
    "green mamba": 200.0,
    "sea snake": 150.0,
    "horned viper": 80.0,
    "diamondback": 100.0,
    "sidewinder": 70.0,
    "trilobite": 50.0,
    "harvestman": 5.0,
    "scorpion": 15.0,
    "black and gold garden spider": 8.0,
    "barn spider": 5.0,
    "garden spider": 5.0,
    "black widow": 20.0,
    "tarantula": 30.0,
    "wolf spider": 10.0,
    "tick": 2.0,
    "centipede": 3.0,
    "black grouse": 100.0,
    "ptarmigan": 80.0,
    "ruffed grouse": 70.0,
    "prairie chicken": 90.0,
    "peacock": 200.0,
    "quail": 40.0,
    "partridge": 60.0,
    "African grey": 1000.0,
    "macaw": 800.0,
    "sulphur-crested cockatoo": 600.0,
    "lorikeet": 400.0,
    "coucal": 150.0,
    "bee eater": 120.0,
    "hornbill": 200.0,
    "hummingbird": 50.0,
    "jacamar": 80.0,
    "toucan": 300.0,
    "drake": 10.0,
    "red-breasted merganser": 50.0,
    "goose": 30.0,
    "black swan": 200.0,
    "tusker": 1000.0,
    "echidna": 500.0,
    "platypus": 800.0,
    "wallaby": 200.0,
    "koala": 500.0,
    "wombat": 300.0,
    "jellyfish": 5.0,
    "sea anemone": 10.0,
    "brain coral": 20.0,
    "flatworm": 2.0,
    "nematode": 1.0,
    "conch": 10.0,
    "snail": 1.0,
    "slug": 1.0,
    "sea slug": 5.0,
    "chiton": 3.0,
    "chambered nautilus": 15.0,
    "Dungeness crab": 8.0,
    "rock crab": 6.0,
    "fiddler crab": 4.0,
    "king crab": 20.0,
    "American lobster": 30.0,
    "spiny lobster": 25.0,
    "crayfish": 10.0,
    "hermit crab": 5.0,
    "isopod": 3.0,
    "white stork": 1000.0,
    "black stork": 800.0,
    "spoonbill": 500.0,
    "flamingo": 400.0,
    "little blue heron": 200.0,
    "American egret": 300.0,
    "bittern": 150.0,
    "crane": 250.0,
    "limpkin": 100.0,
    "European gallinule": 120.0,
    "American coot": 80.0,
    "bustard": 500.0,
    "ruddy turnstone": 30.0,
    "red-backed sandpiper": 25.0,
    "redshank": 20.0,
    "dowitcher": 25.0,
    "oystercatcher": 40.0,
    "pelican": 200.0,
    "king penguin": 150.0,
    "albatross": 300.0,
    "grey whale": 5000.0,
    "killer whale": 10000.0,
    "dugong": 8000.0,
    "sea lion": 2000.0,
    "Chihuahua": 500.0,
    "Japanese spaniel": 300.0,
    "Maltese dog": 400.0,
    "Pekinese": 300.0,
    "Shih-Tzu": 350.0,
    "Blenheim spaniel": 450.0,
    "papillon": 400.0,
    "toy terrier": 250.0,
    "Rhodesian ridgeback": 800.0,
    "Afghan hound": 1000.0,
    "basset": 600.0,
    "beagle": 500.0,
    "bloodhound": 700.0,
    "bluetick": 600.0,
    "black-and-tan coonhound": 550.0,
    "Walker hound": 550.0,
    "English foxhound": 600.0,
    "redbone": 550.0,
    "borzoi": 1000.0,
    "Irish wolfhound": 1200.0,
    "Italian greyhound": 400.0,
    "whippet": 500.0,
    "Ibizan hound": 700.0,
    "Norwegian elkhound": 600.0,
    "otterhound": 800.0,
    "Saluki": 900.0,
    "Scottish deerhound": 1000.0,
    "Weimaraner": 800.0,
    "Staffordshire bullterrier": 600.0,
    "American Staffordshire terrier": 600.0,
    "Bedlington terrier": 700.0,
    "Border terrier": 600.0,
    "Kerry blue terrier": 700.0,
    "Irish terrier": 650.0,
    "Norfolk terrier": 600.0,
    "Norwich terrier": 600.0,
    "Yorkshire terrier": 550.0,
    "wire-haired fox terrier": 550.0,
    "Lakeland terrier": 600.0,
    "Sealyham terrier": 650.0,
    "Airedale": 700.0,
    "cairn": 600.0,
    "Australian terrier": 550.0,
    "Dandie Dinmont": 600.0,
    "Boston bull": 500.0,
    "miniature schnauzer": 550.0,
    "giant schnauzer": 900.0,
    "standard schnauzer": 800.0,
    "Scotch terrier": 600.0,
    "Tibetan terrier": 700.0,
    "silky terrier": 600.0,
    "soft-coated wheaten terrier": 700.0,
    "West Highland white terrier": 650.0,
    "Lhasa": 600.0,
    "flat-coated retriever": 800.0,
    "curly-coated retriever": 800.0,
    "golden retriever": 900.0,
    "Labrador retriever": 800.0,
    "Chesapeake Bay retriever": 900.0,
    "German short-haired pointer": 700.0,
    "vizsla": 700.0,
    "English setter": 800.0,
    "Irish setter": 900.0,
    "Gordon setter": 900.0,
    "Brittany spaniel": 700.0,
    "clumber": 800.0,
    "English springer": 800.0,
    "Welsh springer spaniel": 800.0,
    "cocker spaniel": 700.0,
    "Sussex spaniel": 800.0,
    "Irish water spaniel": 900.0,
    "kuvasz": 1000.0,
    "schipperke": 500.0,
    "groenendael": 900.0,
    "malinois": 900.0,
    "briard": 1000.0,
    "kelpie": 800.0,
    "komondor": 1000.0,
    "Old English sheepdog": 900.0,
    "Shetland sheepdog": 700.0,
    "collie": 800.0,
    "Border collie": 800.0,
    "Bouvier des Flandres": 900.0,
    "Rottweiler": 900.0,
    "German shepherd": 1000.0,
    "Doberman": 900.0,
    "miniature pinscher": 600.0,
    "Greater Swiss Mountain dog": 1000.0,
    "Bernese mountain dog": 1000.0,
    "Appenzeller": 900.0,
    "EntleBucher": 900.0,
    "boxer": 700.0,
    "bull mastiff": 900.0,
    "Tibetan mastiff": 1000.0,
    "French bulldog": 800.0,
    "Great Dane": 1000.0,
    "Saint Bernard": 1000.0,
    "Eskimo dog": 800.0,
    "malamute": 900.0,
    "Siberian husky": 900.0,
    "dalmatian": 700.0,
    "affenpinscher": 600.0,
    "basenji": 600.0,
    "pug": 500.0,
    "Leonberg": 1000.0,
    "Newfoundland": 1000.0,
    "Great Pyrenees": 1000.0,
    "Samoyed": 900.0,
    "Pomeranian": 700.0,
    "chow": 800.0,
    "keeshond": 800.0,
    "Brabancon griffon": 700.0,
    "Pembroke": 800.0,
    "Cardigan": 800.0,
    "toy poodle": 600.0,
    "miniature poodle": 600.0,
    "standard poodle": 700.0,
    "Mexican hairless": 500.0,
    "timber wolf": 5000.0,
    "white wolf": 4000.0,
    "red wolf": 4000.0,
    "coyote": 800.0,
    "dingo": 700.0,
    "dhole": 700.0,
    "African hunting dog": 900.0,
    "hyena": 800.0,
    "red fox": 500.0,
    "kit fox": 400.0,
    "Arctic fox": 600.0,
    "grey fox": 500.0,
    "tabby": 50.0,
    "tiger cat": 50.0,
    "Persian cat": 100.0,
    "Siamese cat": 80.0,
    "Egyptian cat": 80.0,
    "cougar": 800.0,
    "lynx": 600.0,
    "leopard": 1000.0,
    "snow leopard": 1000.0,
    "jaguar": 1200.0,
    "lion": 1500.0,
    "tiger": 1500.0,
    "cheetah": 1200.0,
    "brown bear": 2000.0,
    "American black bear": 1500.0,
    "ice bear": 2500.0,
    "sloth bear": 1800.0,
    "mongoose": 200.0,
    "meerkat": 150.0,
    "tiger beetle": 10.0,
    "ladybug": 5.0,
    "ground beetle": 5.0,
    "long-horned beetle": 5.0,
    "leaf beetle": 5.0,
    "dung beetle": 5.0,
    "rhinoceros beetle": 10.0,
    "weevil": 5.0,
    "fly": 1.0,
    "bee": 1.0,
    "ant": 1.0,
    "grasshopper": 1.0,
    "cricket": 1.0,
    "walking stick": 3.0,
    "cockroach": 1.0,
    "mantis": 3.0,
    "cicada": 2.0,
    "leafhopper": 2.0,
    "lacewing": 2.0,
    "dragonfly": 3.0,
    "damselfly": 3.0,
    "admiral": 3.0,
    "ringlet": 2.0,
    "monarch": 3.0,
    "cabbage butterfly": 3.0,
    "sulphur butterfly": 2.0,
    "lycaenid": 2.0,
    "starfish": 5.0,
    "sea urchin": 5.0,
    "sea cucumber": 5.0,
    "wood rabbit": 10.0,
    "hare": 8.0,
    "Angora": 50.0,
    "hamster": 15.0,
    "porcupine": 30.0,
    "fox squirrel": 10.0,
    "marmot": 20.0,
    "beaver": 40.0,
    "guinea pig": 15.0,
    "sorrel": 30.0,
    "zebra": 1000.0,
    "hog": 300.0,
    "wild boar": 400.0,
    "warthog": 400.0,
    "hippopotamus": 2000.0,
    "ox": 500.0,
    "water buffalo": 1000.0,
    "bison": 1200.0,
    "ram": 300.0,
    "bighorn": 500.0,
    "ibex": 600.0,
    "hartebeest": 800.0,
    "impala": 500.0,
    "gazelle": 400.0,
    "Arabian camel": 1000.0,
    "llama": 800.0,
    "weasel": 100.0,
    "mink": 150.0,
    "polecat": 100.0,
    "black-footed ferret": 200.0,
    "otter": 300.0,
    "skunk": 200.0,
    "badger": 250.0,
    "armadillo": 300.0,
    "three-toed sloth": 500.0,
    "orangutan": 1000.0,
    "gorilla": 1500.0,
    "chimpanzee": 1000.0,
    "gibbon": 800.0,
    "siamang": 800.0,
    "guenon": 300.0,
    "patas": 400.0,
    "baboon": 500.0,
    "macaque": 400.0,
    "langur": 300.0,
    "colobus": 400.0,
    "proboscis monkey": 800.0,
    "marmoset": 300.0,
    "capuchin": 400.0,
    "howler monkey": 600.0,
    "titi": 400.0,
    "spider monkey": 500.0,
    "squirrel monkey": 400.0,
    "Madagascar cat": 500.0,
    "indri": 800.0,
    "Indian elephant": 5000.0,
    "African elephant": 6000.0,
    "lesser panda": 1000.0,
    "giant panda": 2000.0,
    "barracouta": 20.0,
    "eel": 10.0,
    "coho": 20.0,
    "rock beauty": 20.0,
    "anemone fish": 10.0,
    "sturgeon": 50.0,
    "gar": 30.0,
    "lionfish": 20.0,
    "puffer": 30.0,
    "abacus": 10.0,
    "abaya": 50.0,
    "academic gown": 80.0,
    "accordion": 100.0,
    "acoustic guitar": 150.0,
    "aircraft carrier": 5000.0,
    "airliner": 10000.0,
    "airship": 8000.0,
    "altar": 300.0,
    "ambulance": 3000.0,
    "amphibian": 500.0,
    "analog clock": 50.0,
    "apiary": 200.0,
    "apron": 30.0,
    "ashcan": 40.0,
    "assault rifle": 1500.0,
    "backpack": 50.0,
    "bakery": 200.0,
    "balance beam": 300.0,
    "balloon": 20.0,
    "ballpoint": 10.0,
    "Band Aid": 5.0,
    "banjo": 150.0,
    "bannister": 80.0,
    "barbell": 100.0,
    "barber chair": 500.0,
    "barbershop": 300.0,
    "barn": 2000.0,
    "barometer": 100.0,
    "barrel": 200.0,
    "barrow": 80.0,
    "baseball": 30.0,
    "basketball": 40.0,
    "bassinet": 100.0,
    "bassoon": 200.0,
    "bathing cap": 20.0,
    "bath towel": 30.0,
    "bathtub": 500.0,
    "beach wagon": 300.0,
    "beacon": 200.0,
    "beaker": 50.0,
    "bearskin": 500.0,
    "beer bottle": 5.0,
    "beer glass": 10.0,
    "bell cote": 150.0,
    "bib": 5.0,
    "bicycle-built-for-two": 100.0,
    "bikini": 50.0,
    "binder": 20.0,
    "binoculars": 50.0,
    "birdhouse": 100.0,
    "boathouse": 1000.0,
    "bobsled": 300.0,
    "bolo tie": 30.0,
    "bonnet": 50.0,
    "bookcase": 300.0,
    "bookshop": 500.0,
    "bottlecap": 2.0,
    "bow": 20.0,
    "bow tie": 10.0,
    "brass": 50.0,
    "brassiere": 30.0,
    "breakwater": 1000.0,
    "breastplate": 200.0,
    "broom": 30.0,
    "bucket": 20.0,
    "buckle": 10.0,
    "bulletproof vest": 500.0,
    "bullet train": 10000.0,
    "butcher shop": 1000.0,
    "cab": 500.0,
    "caldron": 100.0,
    "candle": 10.0,
    "cannon": 1000.0,
    "canoe": 500.0,
    "can opener": 10.0,
    "cardigan": 100.0,
    "car mirror": 50.0,
    "carousel": 2000.0,
    "carpenter's kit": 200.0,
    "carton": 5.0,
    "car wheel": 50.0,
    "cash machine": 2000.0,
    "cassette": 10.0,
    "cassette player": 100.0,
    "castle": 5000.0,
    "catamaran": 1000.0,
    "CD player": 150.0,
    "cello": 500.0,
    "cellular telephone": 200.0,
    "chain": 20.0,
    "chainlink fence": 200.0,
    "chain mail": 300.0,
    "chain saw": 200.0,
    "chest": 300.0,
    "chiffonier": 200.0,
    "chime": 50.0,
    "china cabinet": 500.0,
    "Christmas stocking": 20.0,
    "church": 2000.0,
    "cinema": 2000.0,
    "cleaver": 50.0,
    "cliff dwelling": 3000.0,
    "cloak": 50.0,
    "clog": 50.0,
    "cocktail shaker": 50.0,
    "coffee mug": 10.0,
    "coffeepot": 30.0,
    "coil": 20.0,
    "combination lock": 30.0,
    "computer keyboard": 50.0,
    "confectionery": 500.0,
    "container ship": 10000.0,
    "convertible": 8000.0,
    "corkscrew": 10.0,
    "cornet": 100.0,
    "cowboy boot": 150.0,
    "cowboy hat": 100.0,
    "cradle": 100.0,
    "crane": 3000.0,
    "crash helmet": 50.0,
    "crate": 30.0,
    "crib": 100.0,
    "Crock Pot": 100.0,
    "croquet ball": 20.0,
    "crutch": 50.0,
    "cuirass": 300.0,
    "dam": 5000.0,
    "desk": 300.0,
    "desktop computer": 1000.0,
    "dial telephone": 100.0,
    "diaper": 10.0,
    "digital clock": 50.0,
    "digital watch": 50.0,
    "dining table": 500.0,
    "dishrag": 5.0,
    "dishwasher": 500.0,
    "disk brake": 50.0,
    "dock": 2000.0,
    "dogsled": 1000.0,
    "dome": 3000.0,
    "doormat": 20.0,
    "drilling platform": 50000.0,
    "drum": 200.0,
    "drumstick": 50.0,
    "dumbbell": 50.0,
    "Dutch oven": 100.0,
    "electric fan": 50.0,
    "electric guitar": 300.0,
    "electric locomotive": 5000.0,
    "entertainment center": 500.0,
    "envelope": 5.0,
    "espresso maker": 50.0,
    "face powder": 10.0,
    "feather boa": 50.0,
    "file": 10.0,
    "fireboat": 1000.0,
    "fire engine": 3000.0,
    "fire screen": 100.0,
    "flagpole": 200.0,
    "flute": 100.0,
    "folding chair": 50.0,
    "football helmet": 200.0,
    "forklift": 3000.0,
    "fountain": 500.0,
    "fountain pen": 50.0,
    "four-poster": 1000.0,
    "freight car": 3000.0,
    "French horn": 300.0,
    "frying pan": 30.0,
    "fur coat": 1000.0,
    "garbage truck": 3000.0,
    "gasmask": 100.0,
    "gas pump": 500.0,
    "goblet": 10.0,
    "go-kart": 1000.0,
    "golf ball": 10.0,
    "golfcart": 1000.0,
    "gondola": 2000.0,
    "gong": 100.0,
    "gown": 200.0,
    "grand piano": 5000.0,
    "greenhouse": 1000.0,
    "grille": 100.0,
    "grocery store": 5000.0,
    "guillotine": 3000.0,
    "hair slide": 10.0,
    "hair spray": 10.0,
    "half track": 5000.0,
    "hammer": 20.0,
    "hamper": 30.0,
    "hand blower": 50.0,
    "hand-held computer": 500.0,
    "handkerchief": 5.0,
    "hard disc": 100.0,
    "harmonica": 50.0,
    "harp": 300.0,
    "harvester": 3000.0,
    "hatchet": 20.0,
    "holster": 50.0,
    "home theater": 500.0,
    "honeycomb": 5.0,
    "hook": 10.0,
    "hoopskirt": 100.0,
    "horizontal bar": 300.0,
    "horse cart": 1000.0,
    "hourglass": 50.0,
    "iPod": 300.0,
    "iron": 30.0,
    "jack-o'-lantern": 20.0,
    "jean": 50.0,
    "jeep": 5000.0,
    "jersey": 100.0,
    "jigsaw puzzle": 30.0,
    "jinrikisha": 1000.0,
    "joystick": 50.0,
    "kimono": 100.0,
    "knee pad": 20.0,
    "knot": 10.0,
    "lab coat": 100.0,
    "ladle": 20.0,
    "lampshade": 30.0,
    "laptop": 1000.0,
    "lawn mower": 500.0,
    "lens cap": 10.0,
    "letter opener": 10.0,
    "library": 2000.0,
    "lifeboat": 1000.0,
    "lighter": 10.0,
    "limousine": 8000.0,
    "liner": 10000.0,
    "lipstick": 10.0,
    "Loafer": 50.0,
    "lotion": 10.0,
    "loudspeaker": 100.0,
    "loupe": 30.0,
    "lumbermill": 3000.0,
    "magnetic compass": 50.0,
    "mailbag": 20.0,
    "mailbox": 50.0,
    "maillot": 100.0,
    "manhole cover": 50.0,
    "maraca": 20.0,
    "marimba": 200.0,
    "mask": 50.0,
    "matchstick": 5.0,
    "maypole": 300.0,
    "maze": 1000.0,
    "measuring cup": 10.0,
    "medicine chest": 200.0,
    "megalith": 5000.0,
    "microphone": 100.0,
    "microwave": 200.0,
    "military uniform": 300.0,
    "milk can": 50.0,
    "minibus": 3000.0,
    "miniskirt": 50.0,
    "minivan": 5000.0,
    "missile": 100000.0,
    "mitten": 10.0,
    "mixing bowl": 20.0,
    "mobile home": 5000.0,
    "Model T": 5000.0,
    "modem": 100.0,
    "monastery": 2000.0,
    "monitor": 500.0,
    "moped": 1000.0,
    "mortar": 100.0,
    "mortarboard": 100.0,
    "mosque": 3000.0,
    "mosquito net": 50.0,
    "motor scooter": 1000.0,
    "mountain bike": 1000.0,
    "mountain tent": 200.0,
    "mouse": 20.0,
    "mousetrap": 20.0,
    "moving van": 3000.0,
    "muzzle": 50.0,
    "nail": 5.0,
    "neck brace": 30.0,
    "necklace": 50.0,
    "nipple": 10.0,
    "notebook": 50.0,
    "obelisk": 2000.0,
    "oboe": 200.0,
    "ocarina": 50.0,
    "odometer": 50.0,
    "oil filter": 10.0,
    "organ": 500.0,
    "oscilloscope": 200.0,
    "overskirt": 50.0,
    "oxcart": 1000.0,
    "oxygen mask": 50.0,
    "packet": 5.0,
    "paddle": 20.0,
    "paddlewheel": 2000.0,
    "padlock": 20.0,
    "paintbrush": 10.0,
    "pajama": 50.0,
    "palace": 5000.0,
    "panpipe": 100.0,
    "paper towel": 5.0,
    "parachute": 1000.0,
    "parallel bars": 300.0,
    "park bench": 300.0,
    "parking meter": 200.0,
    "passenger car": 5000.0,
    "patio": 1000.0,
    "pay-phone": 200.0,
    "pedestal": 100.0,
    "pencil box": 20.0,
    "pencil sharpener": 10.0,
    "perfume": 30.0,
    "Petri dish": 10.0,
    "photocopier": 500.0,
    "pick": 10.0,
    "pickelhaube": 50.0,
    "picket fence": 300.0,
    "pickup": 5000.0,
    "pier": 2000.0,
    "piggy bank": 10.0,
    "pill bottle": 5.0,
    "pillow": 30.0,
    "ping-pong ball": 5.0,
    "pinwheel": 10.0,
    "pirate": 3000.0,
    "pitcher": 20.0,
    "plane": 10000.0,
    "planetarium": 2000.0,
    "plastic bag": 5.0,
    "plate rack": 30.0,
    "plow": 1000.0,
    "plunger": 10.0,
    "Polaroid camera": 200.0,
    "pole": 20.0,
    "police van": 5000.0,
    "poncho": 50.0,
    "pool table": 1000.0,
    "pop bottle": 2.0,
    "pot": 30.0,
    "potter's wheel": 100.0,
    "power drill": 50.0,
    "prayer rug": 50.0,
    "printer": 200.0,
    "prison": 5000.0,
    "projectile": 500.0,
    "projector": 200.0,
    "puck": 10.0,
    "punching bag": 100.0,
    "purse": 50.0,
    "quill": 10.0,
    "quilt": 50.0,
    "racer": 1000.0,
    "racket": 30.0,
    "radiator": 50.0,
    "radio": 100.0,
    "radio telescope": 5000.0,
    "rain barrel": 50.0,
    "recreational vehicle": 5000.0,
    "reel": 50.0,
    "reflex camera": 200.0,
    "refrigerator": 1000.0,
    "remote control": 50.0,
    "restaurant": 5000.0,
    "revolver": 300.0,
    "rifle": 300.0,
    "rocking chair": 200.0,
    "rotisserie": 100.0,
    "rubber eraser": 5.0,
    "rugby ball": 20.0,
    "rule": 5.0,
    "running shoe": 50.0,
    "safe": 300.0,
    "safety pin": 5.0,
    "saltshaker": 10.0,
    "sandal": 50.0,
    "sarong": 50.0,
    "sax": 200.0,
    "scabbard": 20.0,
    "scale": 20.0,
    "school bus": 5000.0,
    "schooner": 5000.0,
    "scoreboard": 200.0,
    "screen": 300.0,
    "screw": 5.0,
    "screwdriver": 10.0,
    "seat belt": 20.0,
    "sewing machine": 200.0,
    "shield": 100.0,
    "shoe shop": 500.0,
    "shoji": 300.0,
    "shopping basket": 30.0,
    "shopping cart": 100.0,
    "shovel": 20.0,
    "shower cap": 10.0,
    "shower curtain": 30.0,
    "ski": 100.0,
    "ski mask": 10.0,
    "sleeping bag": 50.0,
    "slide rule": 50.0,
    "sliding door": 200.0,
    "slot": 100.0,
    "snorkel": 20.0,
    "snowmobile": 1000.0,
    "snowplow": 5000.0,
    "soap dispenser": 20.0,
    "soccer ball": 20.0,
    "sock": 5.0,
    "solar dish": 1000.0,
    "sombrero": 50.0,
    "soup bowl": 10.0,
    "space bar": 20.0,
    "space heater": 50.0,
    "space shuttle": 10000000.0,
    "spatula": 10.0,
    "speedboat": 5000.0,
    "spider web": 5.0,
    "spindle": 50.0,
    "sports car": 10000.0,
    "spotlight": 100.0,
    "stage": 2000.0,
    "steam locomotive": 5000.0,
    "steel arch bridge": 10000.0,
    "steel drum": 100.0,
    "stethoscope": 50.0,
    "stole": 100.0,
    "stone wall": 3000.0,
    "stopwatch": 30.0,
    "stove": 500.0,
    "strainer": 10.0,
    "streetcar": 3000.0,
    "stretcher": 100.0,
    "studio couch": 500.0,
    "stupa": 2000.0,
    "submarine": 5000000.0,
    "suit": 500.0,
    "sundial": 200.0,
    "sunglass": 20.0,
    "sunglasses": 20.0,
    "sunscreen": 10.0,
    "suspension bridge": 10000.0,
    "swab": 10.0,
    "sweatshirt": 50.0,
    "swimming trunks": 50.0,
    "swing": 100.0,
    "switch": 10.0,
    "syringe": 20.0,
    "table lamp": 50.0,
    "tank": 1000000.0,
    "tape player": 100.0,
    "teapot": 30.0,
    "teddy": 50.0,
    "television": 500.0,
    "tennis ball": 10.0,
    "thatch": 100.0,
    "theater curtain": 2000.0,
    "thimble": 5.0,
    "thresher": 3000.0,
    "throne": 500.0,
    "tile roof": 3000.0,
    "toaster": 30.0,
    "tobacco shop": 500.0,
    "toilet seat": 20.0,
    "torch": 20.0,
    "totem pole": 1000.0,
    "tow truck": 5000.0,
    "toyshop": 500.0,
    "tractor": 5000.0,
    "trailer truck": 10000.0,
    "tray": 10.0,
    "trench coat": 100.0,
    "tricycle": 200.0,
    "trimaran": 10000.0,
    "tripod": 50.0,
    "triumphal arch": 2000.0,
    "trolleybus": 5000.0,
    "trombone": 300.0,
    "tub": 100.0,
    "turnstile": 100.0,
    "typewriter keyboard": 50.0,
    "umbrella": 30.0,
    "unicycle": 300.0,
    "upright": 500.0,
    "vacuum": 100.0,
    "vase": 30.0,
    "vault": 5000.0,
    "velvet": 100.0,
    "vending machine": 1000.0,
    "vestment": 100.0,
    "viaduct": 10000.0,
    "violin": 300.0,
    "volleyball": 20.0,
    "waffle iron": 30.0,
    "wall clock": 50.0,
    "wallet": 30.0,
    "wardrobe": 200.0,
    "warplane": 10000000.0,
    "washbasin": 50.0,
    "washer": 500.0,
    "water bottle": 10.0,
    "water jug": 20.0,
    "water tower": 2000.0,
    "whiskey jug": 20.0,
    "whistle": 10.0,
    "wig": 50.0,
    "window screen": 50.0,
    "window shade": 30.0,
    "Windsor tie": 20.0,
    "wine bottle": 20.0,
    "wing": 300.0,
    "wok": 30.0,
    "wooden spoon": 10.0,
    "wool": 100.0,
    "worm fence": 300.0,
    "wreck": 5000.0,
    "yawl": 5000.0,
    "yurt": 10000.0,
    "web site": 500.0,
    "comic book": 10.0,
    "crossword puzzle": 10.0,
    "street sign": 50.0,
    "traffic light": 100.0,
    "book jacket": 10.0,
    "menu": 5.0,
    "plate": 5.0,
    "guacamole": 5.0,
    "consomme": 10.0,
    "hot pot": 20.0,
    "trifle": 30.0,
    "ice cream": 10.0,
    "ice lolly": 5.0,
    "French loaf": 5.0,
    "bagel": 5.0,
    "pretzel": 5.0,
    "cheeseburger": 10.0,
    "hotdog": 5.0,
    "mashed potato": 10.0,
    "head cabbage": 5.0,
    "broccoli": 5.0,
    "cauliflower": 5.0,
    "zucchini": 5.0,
    "spaghetti squash": 5.0,
    "acorn squash": 5.0,
    "butternut squash": 5.0,
    "cucumber": 5.0,
    "artichoke": 10.0,
    "bell pepper": 5.0,
    "cardoon": 10.0,
    "mushroom": 5.0,
    "Granny Smith": 5.0,
    "strawberry": 5.0,
    "orange": 5.0,
    "lemon": 5.0,
    "fig": 5.0,
    "pineapple": 5.0,
    "banana": 5.0,
    "jackfruit": 10.0,
    "custard apple": 10.0,
    "pomegranate": 5.0,
    "hay": 5.0,
    "carbonara": 10.0,
    "chocolate sauce": 10.0,
    "dough": 10.0,
    "meat loaf": 10.0,
    "pizza": 10.0,
    "potpie": 10.0,
    "burrito": 10.0,
    "red wine": 20.0,
    "espresso": 5.0,
    "cup": 5.0,
    "eggnog": 5.0,
    "alp": 1000.0,
    "bubble": 10.0,
    "cliff": 2000.0,
    "coral reef": 1000.0,
    "geyser": 10000.0,
    "lakeside": 2000.0,
    "promontory": 3000.0,
    "sandbar": 500.0,
    "seashore": 2000.0,
    "valley": 2000.0,
    "volcano": 100000.0,
    "ballplayer": 200.0,
    "groom": 200.0,
    "scuba diver": 200.0,
    "rapeseed": 5.0,
    "daisy": 5.0,
    "yellow lady's slipper": 10.0,
    "corn": 5.0,
    "acorn": 5.0,
    "hip": 5.0,
    "buckeye": 5.0,
    "coral fungus": 10.0,
    "agaric": 10.0,
    "gyromitra": 10.0,
    "stinkhorn": 10.0,
    "earthstar": 10.0,
    "hen-of-the-woods": 10.0,
    "bolete": 10.0,
    "ear": 5.0,
    "toilet tissue": 5.0,
  };

}




class BoundingBoxPainter extends CustomPainter {
  final List detections;

  BoundingBoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    for (var detection in detections) {
      final paint = Paint()
        ..color = Colors.red
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      final rect = Rect.fromLTWH(
        detection['xmin'] * size.width,
        detection['ymin'] * size.height,
        (detection['xmax'] - detection['xmin']) * size.width,
        (detection['ymax'] - detection['ymin']) * size.height,
      );

      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;

}




class AverageGeneratedItemData {
  final String name;
  final double price;
  final ranking;

  AverageGeneratedItemData({required this.name,required this.price, required this.ranking});

}


// img.Image _convertYUV420toImageColor(CameraImage image) {
//   var imgLib = img.Image(image.width, image.height); // Create Image buffer
//
//   Plane plane = image.planes[0];
//   const int shift = (0xFF << 24);
//
//   for (int y = 0; y < image.height; y++) {
//     for (int x = 0; x < image.width; x++) {
//       int pixelIndex = y * image.width + x;
//       if (pixelIndex < plane.bytes.length) {
//         final int pixelColor = plane.bytes[pixelIndex];
//         final int pixelColorUp = pixelColor << 24;
//         imgLib.data[pixelIndex] = shift | pixelColorUp;
//       }
//     }
//   }
//
//   return imgLib;
// }
//
//
// void processCameraImage(CameraImage? cameraImage) async {
//   // Check if cameraImage is not null
//   if (cameraImage == null) {
//     print('cameraImage is null');
//     return;
//   }
//   // Print camera image details for debugging
//   // print('Received camera image - width: ${cameraImage.width}, height: ${cameraImage.height}, format: ${cameraImage.format}');
//   //The image captured by the camera is converted into a byte array which is then resized and normalized (if necessary) to match the input size expected by the model.
//   //convert jpeg to byte array for model
//   final img.Image convertedImage = img.decodeImage(cameraImage.planes[0].bytes)!;
//   // final List<double> normalizedImage = convertedImage.data.buffer.asFloat32List();
//   image = convertedImage;
//   //print('image: $image');
//   // Assign the processed image to latestImage
//   latestImage = image;
//   // Process image
//   await processImage();
//   setState(() {});
// }



















// Process picked image
// Future<void> processImage() async {
//   // Resize the image to the input size of the model
//   img.Image resizedImage = img.copyResize(image!, width: 224, height: 224);
//
//   // Convert the resized image to a byte list
//   List<int> byteList = resizedImage.getBytes();
//
//   // Preprocess the image bytes if necessary
//   // ...
//
//
//
//   //what should this be?
//   //imagePath =
//
//   // Create the input tensor
//   // inputTensor = Tensor.fromList(TfLiteType.uint8, [1, 224, 224, 3], byteList);
//   if (imagePath != null) {
//     // Read image bytes from file
//     final imageData = File(imagePath!).readAsBytesSync();
//
//     // Decode image using package:image/image.dart (https://pub.dev/image)
//     image = img.decodeImage(imageData);
//     setState(() {});
//
//     // Resize image for model input (Mobilenet use [224, 224])
//     final imageInput = img.copyResize(
//       image!,
//       width: 224,
//       height: 224,
//     );
//
//     // Get image matrix representation [224, 224, 3]
//     final imageMatrix = List.generate(
//       imageInput.height,
//           (y) => List.generate(
//         imageInput.width,
//             (x) {
//           final pixel = imageInput.getPixel(x, y);
//           return [pixel >> 16 & 0xFF, pixel >> 8 & 0xFF, pixel & 0xFF];
//         },
//       ),
//     );
//
//     // Run model inference
//     debugPrint('Running inference on image $imagePath');
//     List<double> modelOutput = await runModelOnFrame(imageMatrix);
//
//     // Extract the bounding boxes from the output
//     boundingBoxes = extractBoundingBoxes(modelOutput);
//
//     // Trigger a rebuild
//     setState(() {});
//   }else{
//     debugPrint('No image path');
//   }
// }
//
// @override
// Widget build(BuildContext context) {
//   if (!controller.value.isInitialized) {
//     return Container();
//   }
//   return Scaffold(
//     appBar: AppBar(
//       title: const Text('Live Object Detection'),
//       backgroundColor: Colors.indigo,
//     ),
//     body: AspectRatio(
//       aspectRatio: controller.value.aspectRatio,
//       child: latestImage == null
//           ? CameraPreview(controller)
//           : Image.memory(Uint8List.fromList(img.encodeJpg(latestImage!))),
//     ),
//   );
// }

// convertCameraImageToImage(CameraImage cameraImage) {
//   try {
//     final int width = cameraImage.planes[0].width!;
//     final int height = cameraImage.planes[0].height!;
//
//     final int uvRowStride = cameraImage.planes[1].bytesPerRow;
//     final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;
//
//     final Uint8List yPlaneBytes = cameraImage.planes[0].bytes;
//     final Uint8List uvPlaneBytes = cameraImage.planes[1].bytes;
//
//     final img.Image rgbImage = img.Image(width, height, channels: img.Channels.rgb);
//
//     for (int x = 0; x < width; x++) {
//       for (int y = 0; y < height; y++) {
//         final int uvIndex = uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
//
//         final int yValue = yPlaneBytes[y * width + x];
//         final int uValue = uvPlaneBytes[uvIndex];
//         final int vValue = uvPlaneBytes[uvIndex + 1];
//
//         final int r = (yValue + 1.402 * (vValue - 128)).round().clamp(0, 255);
//         final int g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128)).round().clamp(0, 255);
//         final int b = (yValue + 1.772 * (uValue - 128)).round().clamp(0, 255);
//
//         rgbImage.setPixelRgba(x, y, r, g, b);
//       }
//     }
//
//     return rgbImage;
//   } catch (e) {
//     print('Error converting camera image to image: $e');
//     return null;
//   }
// }

// Check if the camera image format matches the expected format (YUV_420_888)
// Check if the camera image format matches the expected format (YUV_420_888)
// if (cameraImage.planes.length < 3 || cameraImage.planes[1].bytesPerRow != 1 || cameraImage.planes[1].bytesPerPixel != 2) {
//   print('Unexpected camera image format');
//   //okay so we don't need to convert the image to rgb because it's already rgb?
//
//   // Set image to the cameraImage
//   image = convertCameraImageToImageV2(cameraImage);
//
//   // Process image
//   await processImage();
//
//   // Trigger a rebuild
//   setState(() {});
// }else{
//
//   // if (cameraImage.format != ImageFormatGroup.yuv420) {
//   //   print('Unexpected camera image format: ${cameraImage.format}');
//   //   return;
//   // }
//   // Preprocessing here
//   // Convert the YUV_420_888 image to RGB
//   img.Image? convertedImage = convertCameraImageToImage(cameraImage);
//   if (convertedImage == null) {
//     print('convertedImage is null');
//     return;
//   }
//   // Set image to the converted image
//   image = convertedImage;
//
//   // Process image
//   await processImage();
//
//   // Trigger a rebuild
//   setState(() {});
// }

// convertCameraImageToImage(CameraImage cameraImage) {
//
//     final int? width = cameraImage.planes[0].width;
//     final int? height = cameraImage.planes[0].height;
//     if (width == null || height == null) {
//       print('Width or height is null. Returning.');
//       return null;
//     }
//
//     final Uint8List yPlane = cameraImage.planes[0].bytes;
//     final Uint8List uPlane = cameraImage.planes[1].bytes;
//     final Uint8List vPlane = cameraImage.planes[2].bytes;
//
//     final int uvRowStride = cameraImage.planes[1].bytesPerRow;
//     final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;
//
//     // Convert the YUV image data to RGB format
//     final img.Image rgbImage = img.Image(width!, height!);
//     try {
//     for (int y = 0; y < height; y++) {
//       for (int x = 0; x < width; x++) {
//         final int uvIndex = uvPixelStride * (x / 2).floor() +
//             uvRowStride * (y / 2).floor();
//
//         final double yp = yPlane[y * width + x].toDouble();
//         final double up = uPlane[uvIndex].toDouble() - 128;
//         final double vp = vPlane[uvIndex].toDouble() - 128;
//
//         // Perform color space conversion
//         int r = (yp + 1.370705 * vp).round().clamp(0, 255);
//         int g = (yp - 0.698001 * vp - 0.337633 * up).round().clamp(0, 255);
//         int b = (yp + 1.732446 * up).round().clamp(0, 255);
//
//         // Set the pixel color
//         rgbImage.setPixelRgba(x, y, r, g, b);
//       }
//     }
//   } catch (e) {
//     print(">>>>>>>>>>>> ERROR:" + e.toString());
//   }
//
//   return rgbImage;
// }
// Future<ui.Image> convertCameraImageToUiImage(CameraImage cameraImage) async {
//   final int width = cameraImage.width;
//   final int height = cameraImage.height;
//
//   // Convert the YUV image to a RGB image.
//   final yuvRows = cameraImage.planes[0].bytesPerRow;
//   final uvRows = cameraImage.planes[1].bytesPerRow;
//   final uvRowStride = cameraImage.planes[1].bytesPerRow;
//   final uvPixelStride = cameraImage.planes[1].bytesPerPixel;
//
//   final image = Uint8List(width * height * 4);
//
//   for (int y = 0; y < height; y++) {
//     for (int x = 0; x < width; x++) {
//       final int uvIndex = uvPixelStride * (x/2).floor() + uvRowStride * (y/2).floor();
//       final int index = y * width + x;
//
//       final yp = cameraImage.planes[0].bytes[y*yuvRows + x];
//       final up = cameraImage.planes[1].bytes[uvIndex];
//       final vp = cameraImage.planes[2].bytes[uvIndex];
//
//       // Calculate pixel color
//       int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
//       int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91).round().clamp(0, 255);
//       int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
//
//       // Store the pixel color in the image data.
//       image[index * 4 + 0] = r;
//       image[index * 4 + 1] = g;
//       image[index * 4 + 2] = b;
//       image[index * 4 + 3] = 255;
//     }
//   }
//   // Create the Image object
//   final Completer<ui.Image> completer = Completer();
//   ui.decodeImageFromPixels(image, width, height, ui.PixelFormat.rgba8888, (ui.Image img) {
//     completer.complete(img);
//   });
//
//   return completer.future;
// }
// img.Image convertCameraImageToImageV2(CameraImage cameraImage) {
//   final int width = cameraImage.width;
//   final int height = cameraImage.height;
//
//   var imgLibImage = img.Image(width, height);  // Create an empty image
//
//   final plane = cameraImage.planes[0];
//   final planeData = plane.bytes;
//
//   final pixelStride = plane.pixelStride;
//   final rowStride = plane.bytesPerRow;
//   final rowPadding = rowStride - (pixelStride * width);
//
//   var pixelIndex = 0;
//   var pixelInBytesIndex = 0;
//   for (var i = 0; i < height; i++) {
//     for (var j = 0; j < width; j++) {
//       final pixelData = planeData[pixelInBytesIndex];
//       imgLibImage.data[pixelIndex++] = pixelData;
//       pixelInBytesIndex += pixelStride;
//     }
//     pixelInBytesIndex += rowPadding;
//   }
//
//   return imgLibImage;
// }
// Future<void> processImage() async {
//   if (imagePath != null) {
//     // Read image bytes from file
//     final imageData = File(imagePath!).readAsBytesSync();
//
//     // Decode image using package:image/image.dart (https://pub.dev/image)
//     image = img.decodeImage(imageData);
//     setState(() {});
//
//     // Resize image for model input (Mobilenet use [224, 224])
//     final imageInput = img.copyResize(
//       image!,
//       width: 224,
//       height: 224,
//     );
//
//     // Get image matrix representation [224, 224, 3]
//     final imageMatrix = List.generate(
//       imageInput.height,
//           (y) => List.generate(
//         imageInput.width,
//             (x) {
//           final pixel = imageInput.getPixel(x, y);
//           return [pixel >> 16 & 0xFF, pixel >> 8 & 0xFF, pixel & 0xFF];
//         },
//       ),
//     );
//
//     // Run model inference
//     runInference(imageMatrix);
//   }
// }

//
// Future<List<double>> runModelOnFrame(CameraImage img) async {
//   // Preprocessing to convert img to the appropriate input format goes here...
//
//   // Prepare the input buffer
//   var inputSize = inputTensor!.shape.reduce((value, element) => value * element);
//   var inputBuffer = List<double>.filled(inputSize, 0);
//
//   // Prepare the output buffer
//   var outputSize = outputTensor!.shape.reduce((value, element) => value * element);
//   var outputBuffer = List<double>.filled(outputSize, 0);
//
//   // Run the model
//   interpreter.run(inputBuffer, outputBuffer);
//
//   // The outputBuffer now contains the output of the model
//   return outputBuffer;
// }


//-----------------------------------------------
// class HomeTensorFlowTest extends StatefulWidget {
//   const HomeTensorFlowTest({super.key});
//
//   @override
//   State<HomeTensorFlowTest> createState() => _HomeState();
// }
//
// class _HomeState extends State<HomeTensorFlowTest> {
//   static const modelPath = 'assets/mobilenet/mobilenet_v1_1.0_224_quant.tflite';
//   static const labelsPath = 'assets/mobilenet/labels.txt';
//
//   late final Interpreter interpreter;
//   late final List<String> labels;
//
//   Tensor? inputTensor;
//   Tensor? outputTensor;
//
//   final imagePicker = ImagePicker();
//   String? imagePath;
//   img.Image? image;
//
//
// // Add a new Map to hold the selected state for each item
//   Map<String, bool> selectionStatus = {};
//   List<AverageGeneratedItemData> item_matched_data_objects = [];
//
//   // Add a String to hold the sentence
//   String sentence = '';
//   String priceDisplay = '';
//   @override
//   void initState() {
//     super.initState();
//     // Load model and labels from assets
//     loadModel();
//     loadLabels();
//   }
//
//   // Clean old results when press some take picture button
//   void cleanResult() {
//     imagePath = null;
//     image = null;
//     // classification = null;
//     setState(() {});
//   }
//
//   // Load model
//   Future<void> loadModel() async {
//     final options = InterpreterOptions();
//
//     // Use XNNPACK Delegate
//     if (Platform.isAndroid) {
//       options.addDelegate(XNNPackDelegate());
//     }
//
//     // Use GPU Delegate
//     // doesn't work on emulator
//     // if (Platform.isAndroid) {
//     //   options.addDelegate(GpuDelegateV2());
//     // }
//
//     // Use Metal Delegate
//     if (Platform.isIOS) {
//       options.addDelegate(GpuDelegate());
//     }
//
//     // Load model from assets
//     interpreter = await Interpreter.fromAsset(modelPath, options: options);
//     // Get tensor input shape [1, 224, 224, 3]
//     inputTensor = interpreter.getInputTensors().first;
//     // Get tensor output shape [1, 1001]
//     outputTensor = interpreter.getOutputTensors().first;
//     setState(() {});
//
//     log('Interpreter loaded successfully');
//   }
//
//   // Load labels from assets
//   Future<void> loadLabels() async {
//     final labelTxt = await rootBundle.loadString(labelsPath);
//     labels = labelTxt.split('\n');
//   }
//
//   // Process picked image
//   Future<void> processImage() async {
//     if (imagePath != null) {
//       // Read image bytes from file
//       final imageData = File(imagePath!).readAsBytesSync();
//
//       // Decode image using package:image/image.dart (https://pub.dev/image)
//       image = img.decodeImage(imageData);
//       setState(() {});
//
//       // Resize image for model input (Mobilenet use [224, 224])
//       final imageInput = img.copyResize(
//         image!,
//         width: 224,
//         height: 224,
//       );
//
//       // Get image matrix representation [224, 224, 3]
//       final imageMatrix = List.generate(
//         imageInput.height,
//             (y) => List.generate(
//           imageInput.width,
//               (x) {
//             final pixel = imageInput.getPixel(x, y);
//             return [pixel >> 16 & 0xFF, pixel >> 8 & 0xFF, pixel & 0xFF];
//           },
//         ),
//       );
//
//       // Run model inference
//       runInference(imageMatrix);
//     }
//   }
//
//   // Run inference
//   Future<void> runInference(
//       List<List<List<num>>> imageMatrix,
//       ) async {
//     // Set tensor input [1, 224, 224, 3]
//     final input = [imageMatrix];
//     // Set tensor output [1, 1001]
//     final output = [List<int>.filled(1001, 0)];
//
//
//
//     // Run inference
//     interpreter.run(input, output);
//
//     // Get first output tensor
//     final result = output.first;
//
//     // Set classification map {label: points}
//     Map<String, int>? classification = <String, int>{};
//
//     for (var i = 0; i < result.length; i++) {
//       if (result[i] != 0) {
//         // Set label: points
//         classification![labels[i]] = result[i];
//         //Generate item data model from matching pricing_data_array
//         //lookup price:
//         double price_result = 0.0;
//         pricing_data_array.forEach((key, value) {
//           if (key.toString().toLowerCase() == labels[i].toString().toLowerCase()) {
//             price_result = value;
//           }
//         });
//         AverageGeneratedItemData itemData = AverageGeneratedItemData(name: labels[i], price: price_result, ranking: result[i]);
//         item_matched_data_objects.add(itemData);
//       }
//     }
//
//     setState(() {});
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor: Colors.white,
//       appBar: AppBar(
//         title: const Text('Object Detection - TensorFlow'),
//         backgroundColor: Colors.indigo,
//       ),
//       body: SingleChildScrollView(
//         child: Container(
//           height: MediaQuery.of(context).size.height, // set height to the desired value
//           width: MediaQuery.of(context).size.width, // set width to the desired value
//           child: Column(
//             mainAxisAlignment: MainAxisAlignment.center,
//             mainAxisSize: MainAxisSize.min,
//             children: [
//               if (imagePath != null) Padding(
//                 padding: const EdgeInsets.all(8.0),
//                 child: Image.file(File(imagePath!)),
//               )
//               else   IconButton(
//                 onPressed: () async {
//                   cleanResult();
//                   final result = await imagePicker.pickImage(
//                     source: ImageSource.gallery,
//                   );
//
//                   imagePath = result?.path;
//                   setState(() {});
//                   processImage();
//                 },
//                 icon: const Icon(
//                   Icons.photo_library_outlined,
//                   size: 32,
//                   color: Colors.indigo,
//                 ),
//               ),
//               SizedBox(height: 10),
//               sentence == ''
//                   ? const SizedBox.shrink()
//                   : Wrap(
//                 children: [
//                   Text(
//                     overflow: TextOverflow.ellipsis,
//                     maxLines: 2,
//                     sentence,
//                     style: const TextStyle(
//                       color: Colors.black,
//                       fontSize: 18,
//                       fontWeight: FontWeight.bold,
//                     ),
//                   ),
//                 ],
//               ),
//               priceDisplay == ''
//                   ? const SizedBox.shrink()
//                   : Wrap(
//                 children: [
//                   Text(
//                     overflow: TextOverflow.ellipsis,
//                     maxLines: 2,
//                     priceDisplay,
//                     style: const TextStyle(
//                       color: Colors.green,
//                       fontSize: 18,
//                       fontWeight: FontWeight.bold,
//                     ),
//                   ),
//                 ],
//               ),
//               SizedBox(height: 10),
//               if (item_matched_data_objects != null || item_matched_data_objects.length > 0)
//                 GridView.builder(
//                   shrinkWrap: true,
//                   gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
//                     crossAxisCount: 3,
//                     childAspectRatio: 3,
//                   ),
//                   itemCount: item_matched_data_objects!.length,
//                   itemBuilder: (context, index) {
//                     final entries = item_matched_data_objects!.toList();
//                     entries.sort((a, b) => a.ranking.compareTo(b.ranking));
//                     final entry = entries.reversed.elementAt(index);
//
//                     //pull current item from AverageGeneratedItemData object
//                     String itemName = entry.name;
//                     double itemPrice = entry.price;
//                     int itemRanking = entry.ranking;
//
//                     //itemPrice should be usd formatted
//                     String formattedPrice = NumberFormat.simpleCurrency(locale: 'en_US').format(itemPrice);
//
//
//                     // Initialize the selection state for this item if it hasn't been done yet
//                     selectionStatus[itemName] ??= false;
//                     if(itemRanking > 1) {
//                       return Padding(
//                         padding: const EdgeInsets.all(3.0),
//                         child: InkWell(
//                           onTap: () {
//                             // Toggle the selected state for this item
//                             setState(() {
//                               selectionStatus[itemName] = !selectionStatus[itemName]!;
//
//                               // Add the item to the sentence if it's selected, otherwise remove it
//                               if (selectionStatus[itemName]!) {
//                                 sentence += '$itemName ';
//                                 priceDisplay = formattedPrice;
//                               } else {
//                                 sentence = sentence.replaceAll('$itemName ', '');
//                                 priceDisplay = "";
//                               }
//                             });
//                           },
//                           child: Material(
//                             color: selectionStatus[itemName]!
//                                 ? Colors.blue.withOpacity(0.3)
//                                 : Colors.transparent,
//                             shape: RoundedRectangleBorder(
//                               borderRadius: BorderRadius.circular(8),
//                               side: BorderSide(
//                                 color: Colors.indigo,
//                                 width: 1,
//                               ),
//                             ),
//                             child: Padding(
//                               padding: EdgeInsets.all(8),
//                               child: Row(
//                                 mainAxisAlignment: MainAxisAlignment.center,
//                                 mainAxisSize: MainAxisSize.min,
//                                 children: [
//                                   Text(
//                                     '$itemName',
//                                     style: TextStyle(color: Colors.blue,
//                                         fontSize: 12,
//                                         fontWeight: FontWeight.bold),
//                                   ),
//                                 ],
//                               ),
//                             ),
//                           ),
//                         ),
//                       );
//                     } else {
//                       return SizedBox.shrink();
//                     }
//                   },
//                 ),
//
//             ],
//           ),
//         ),
//       ),
//     );
//   }
//
//   Map pricing_data_array = {
//     "background": 0,
//     "tench": 10.0,
//     "goldfish": 5.0,
//     "great white shark": 10000.0,
//     "tiger shark": 8000.0,
//     "hammerhead": 6000.0,
//     "electric ray": 20.0,
//     "stingray": 30.0,
//     "cock": 5.0,
//     "hen": 5.0,
//     "ostrich": 500.0,
//     "brambling": 15.0,
//     "goldfinch": 10.0,
//     "house finch": 8.0,
//     "junco": 7.0,
//     "indigo bunting": 20.0,
//     "robin": 10.0,
//     "bulbul": 25.0,
//     "jay": 15.0,
//     "magpie": 20.0,
//     "chickadee": 10.0,
//     "water ouzel": 30.0,
//     "kite": 50.0,
//     "bald eagle": 1000.0,
//     "vulture": 200.0,
//     "great grey owl": 500.0,
//     "European fire salamander": 50.0,
//     "common newt": 10.0,
//     "eft": 10.0,
//     "spotted salamander": 20.0,
//     "axolotl": 30.0,
//     "bullfrog": 15.0,
//     "tree frog": 10.0,
//     "tailed frog": 20.0,
//     "loggerhead": 500.0,
//     "leatherback turtle": 1000.0,
//     "mud turtle": 20.0,
//     "terrapin": 30.0,
//     "box turtle": 50.0,
//     "banded gecko": 40.0,
//     "common iguana": 200.0,
//     "American chameleon": 150.0,
//     "whiptail": 30.0,
//     "agama": 25.0,
//     "frilled lizard": 100.0,
//     "alligator lizard": 50.0,
//     "Gila monster": 300.0,
//     "green lizard": 20.0,
//     "African chameleon": 150.0,
//     "Komodo dragon": 5000.0,
//     "African crocodile": 2000.0,
//     "American alligator": 1500.0,
//     "triceratops": 1000000.0,
//     "thunder snake": 20.0,
//     "ringneck snake": 10.0,
//     "hognose snake": 30.0,
//     "green snake": 10.0,
//     "king snake": 50.0,
//     "garter snake": 15.0,
//     "water snake": 20.0,
//     "vine snake": 25.0,
//     "night snake": 15.0,
//     "boa constrictor": 200.0,
//     "rock python": 300.0,
//     "Indian cobra": 100.0,
//     "green mamba": 200.0,
//     "sea snake": 150.0,
//     "horned viper": 80.0,
//     "diamondback": 100.0,
//     "sidewinder": 70.0,
//     "trilobite": 50.0,
//     "harvestman": 5.0,
//     "scorpion": 15.0,
//     "black and gold garden spider": 8.0,
//     "barn spider": 5.0,
//     "garden spider": 5.0,
//     "black widow": 20.0,
//     "tarantula": 30.0,
//     "wolf spider": 10.0,
//     "tick": 2.0,
//     "centipede": 3.0,
//     "black grouse": 100.0,
//     "ptarmigan": 80.0,
//     "ruffed grouse": 70.0,
//     "prairie chicken": 90.0,
//     "peacock": 200.0,
//     "quail": 40.0,
//     "partridge": 60.0,
//     "African grey": 1000.0,
//     "macaw": 800.0,
//     "sulphur-crested cockatoo": 600.0,
//     "lorikeet": 400.0,
//     "coucal": 150.0,
//     "bee eater": 120.0,
//     "hornbill": 200.0,
//     "hummingbird": 50.0,
//     "jacamar": 80.0,
//     "toucan": 300.0,
//     "drake": 10.0,
//     "red-breasted merganser": 50.0,
//     "goose": 30.0,
//     "black swan": 200.0,
//     "tusker": 1000.0,
//     "echidna": 500.0,
//     "platypus": 800.0,
//     "wallaby": 200.0,
//     "koala": 500.0,
//     "wombat": 300.0,
//     "jellyfish": 5.0,
//     "sea anemone": 10.0,
//     "brain coral": 20.0,
//     "flatworm": 2.0,
//     "nematode": 1.0,
//     "conch": 10.0,
//     "snail": 1.0,
//     "slug": 1.0,
//     "sea slug": 5.0,
//     "chiton": 3.0,
//     "chambered nautilus": 15.0,
//     "Dungeness crab": 8.0,
//     "rock crab": 6.0,
//     "fiddler crab": 4.0,
//     "king crab": 20.0,
//     "American lobster": 30.0,
//     "spiny lobster": 25.0,
//     "crayfish": 10.0,
//     "hermit crab": 5.0,
//     "isopod": 3.0,
//     "white stork": 1000.0,
//     "black stork": 800.0,
//     "spoonbill": 500.0,
//     "flamingo": 400.0,
//     "little blue heron": 200.0,
//     "American egret": 300.0,
//     "bittern": 150.0,
//     "crane": 250.0,
//     "limpkin": 100.0,
//     "European gallinule": 120.0,
//     "American coot": 80.0,
//     "bustard": 500.0,
//     "ruddy turnstone": 30.0,
//     "red-backed sandpiper": 25.0,
//     "redshank": 20.0,
//     "dowitcher": 25.0,
//     "oystercatcher": 40.0,
//     "pelican": 200.0,
//     "king penguin": 150.0,
//     "albatross": 300.0,
//     "grey whale": 5000.0,
//     "killer whale": 10000.0,
//     "dugong": 8000.0,
//     "sea lion": 2000.0,
//     "Chihuahua": 500.0,
//     "Japanese spaniel": 300.0,
//     "Maltese dog": 400.0,
//     "Pekinese": 300.0,
//     "Shih-Tzu": 350.0,
//     "Blenheim spaniel": 450.0,
//     "papillon": 400.0,
//     "toy terrier": 250.0,
//     "Rhodesian ridgeback": 800.0,
//     "Afghan hound": 1000.0,
//     "basset": 600.0,
//     "beagle": 500.0,
//     "bloodhound": 700.0,
//     "bluetick": 600.0,
//     "black-and-tan coonhound": 550.0,
//     "Walker hound": 550.0,
//     "English foxhound": 600.0,
//     "redbone": 550.0,
//     "borzoi": 1000.0,
//     "Irish wolfhound": 1200.0,
//     "Italian greyhound": 400.0,
//     "whippet": 500.0,
//     "Ibizan hound": 700.0,
//     "Norwegian elkhound": 600.0,
//     "otterhound": 800.0,
//     "Saluki": 900.0,
//     "Scottish deerhound": 1000.0,
//     "Weimaraner": 800.0,
//     "Staffordshire bullterrier": 600.0,
//     "American Staffordshire terrier": 600.0,
//     "Bedlington terrier": 700.0,
//     "Border terrier": 600.0,
//     "Kerry blue terrier": 700.0,
//     "Irish terrier": 650.0,
//     "Norfolk terrier": 600.0,
//     "Norwich terrier": 600.0,
//     "Yorkshire terrier": 550.0,
//     "wire-haired fox terrier": 550.0,
//     "Lakeland terrier": 600.0,
//     "Sealyham terrier": 650.0,
//     "Airedale": 700.0,
//     "cairn": 600.0,
//     "Australian terrier": 550.0,
//     "Dandie Dinmont": 600.0,
//     "Boston bull": 500.0,
//     "miniature schnauzer": 550.0,
//     "giant schnauzer": 900.0,
//     "standard schnauzer": 800.0,
//     "Scotch terrier": 600.0,
//     "Tibetan terrier": 700.0,
//     "silky terrier": 600.0,
//     "soft-coated wheaten terrier": 700.0,
//     "West Highland white terrier": 650.0,
//     "Lhasa": 600.0,
//     "flat-coated retriever": 800.0,
//     "curly-coated retriever": 800.0,
//     "golden retriever": 900.0,
//     "Labrador retriever": 800.0,
//     "Chesapeake Bay retriever": 900.0,
//     "German short-haired pointer": 700.0,
//     "vizsla": 700.0,
//     "English setter": 800.0,
//     "Irish setter": 900.0,
//     "Gordon setter": 900.0,
//     "Brittany spaniel": 700.0,
//     "clumber": 800.0,
//     "English springer": 800.0,
//     "Welsh springer spaniel": 800.0,
//     "cocker spaniel": 700.0,
//     "Sussex spaniel": 800.0,
//     "Irish water spaniel": 900.0,
//     "kuvasz": 1000.0,
//     "schipperke": 500.0,
//     "groenendael": 900.0,
//     "malinois": 900.0,
//     "briard": 1000.0,
//     "kelpie": 800.0,
//     "komondor": 1000.0,
//     "Old English sheepdog": 900.0,
//     "Shetland sheepdog": 700.0,
//     "collie": 800.0,
//     "Border collie": 800.0,
//     "Bouvier des Flandres": 900.0,
//     "Rottweiler": 900.0,
//     "German shepherd": 1000.0,
//     "Doberman": 900.0,
//     "miniature pinscher": 600.0,
//     "Greater Swiss Mountain dog": 1000.0,
//     "Bernese mountain dog": 1000.0,
//     "Appenzeller": 900.0,
//     "EntleBucher": 900.0,
//     "boxer": 700.0,
//     "bull mastiff": 900.0,
//     "Tibetan mastiff": 1000.0,
//     "French bulldog": 800.0,
//     "Great Dane": 1000.0,
//     "Saint Bernard": 1000.0,
//     "Eskimo dog": 800.0,
//     "malamute": 900.0,
//     "Siberian husky": 900.0,
//     "dalmatian": 700.0,
//     "affenpinscher": 600.0,
//     "basenji": 600.0,
//     "pug": 500.0,
//     "Leonberg": 1000.0,
//     "Newfoundland": 1000.0,
//     "Great Pyrenees": 1000.0,
//     "Samoyed": 900.0,
//     "Pomeranian": 700.0,
//     "chow": 800.0,
//     "keeshond": 800.0,
//     "Brabancon griffon": 700.0,
//     "Pembroke": 800.0,
//     "Cardigan": 800.0,
//     "toy poodle": 600.0,
//     "miniature poodle": 600.0,
//     "standard poodle": 700.0,
//     "Mexican hairless": 500.0,
//     "timber wolf": 5000.0,
//     "white wolf": 4000.0,
//     "red wolf": 4000.0,
//     "coyote": 800.0,
//     "dingo": 700.0,
//     "dhole": 700.0,
//     "African hunting dog": 900.0,
//     "hyena": 800.0,
//     "red fox": 500.0,
//     "kit fox": 400.0,
//     "Arctic fox": 600.0,
//     "grey fox": 500.0,
//     "tabby": 50.0,
//     "tiger cat": 50.0,
//     "Persian cat": 100.0,
//     "Siamese cat": 80.0,
//     "Egyptian cat": 80.0,
//     "cougar": 800.0,
//     "lynx": 600.0,
//     "leopard": 1000.0,
//     "snow leopard": 1000.0,
//     "jaguar": 1200.0,
//     "lion": 1500.0,
//     "tiger": 1500.0,
//     "cheetah": 1200.0,
//     "brown bear": 2000.0,
//     "American black bear": 1500.0,
//     "ice bear": 2500.0,
//     "sloth bear": 1800.0,
//     "mongoose": 200.0,
//     "meerkat": 150.0,
//     "tiger beetle": 10.0,
//     "ladybug": 5.0,
//     "ground beetle": 5.0,
//     "long-horned beetle": 5.0,
//     "leaf beetle": 5.0,
//     "dung beetle": 5.0,
//     "rhinoceros beetle": 10.0,
//     "weevil": 5.0,
//     "fly": 1.0,
//     "bee": 1.0,
//     "ant": 1.0,
//     "grasshopper": 1.0,
//     "cricket": 1.0,
//     "walking stick": 3.0,
//     "cockroach": 1.0,
//     "mantis": 3.0,
//     "cicada": 2.0,
//     "leafhopper": 2.0,
//     "lacewing": 2.0,
//     "dragonfly": 3.0,
//     "damselfly": 3.0,
//     "admiral": 3.0,
//     "ringlet": 2.0,
//     "monarch": 3.0,
//     "cabbage butterfly": 3.0,
//     "sulphur butterfly": 2.0,
//     "lycaenid": 2.0,
//     "starfish": 5.0,
//     "sea urchin": 5.0,
//     "sea cucumber": 5.0,
//     "wood rabbit": 10.0,
//     "hare": 8.0,
//     "Angora": 50.0,
//     "hamster": 15.0,
//     "porcupine": 30.0,
//     "fox squirrel": 10.0,
//     "marmot": 20.0,
//     "beaver": 40.0,
//     "guinea pig": 15.0,
//     "sorrel": 30.0,
//     "zebra": 1000.0,
//     "hog": 300.0,
//     "wild boar": 400.0,
//     "warthog": 400.0,
//     "hippopotamus": 2000.0,
//     "ox": 500.0,
//     "water buffalo": 1000.0,
//     "bison": 1200.0,
//     "ram": 300.0,
//     "bighorn": 500.0,
//     "ibex": 600.0,
//     "hartebeest": 800.0,
//     "impala": 500.0,
//     "gazelle": 400.0,
//     "Arabian camel": 1000.0,
//     "llama": 800.0,
//     "weasel": 100.0,
//     "mink": 150.0,
//     "polecat": 100.0,
//     "black-footed ferret": 200.0,
//     "otter": 300.0,
//     "skunk": 200.0,
//     "badger": 250.0,
//     "armadillo": 300.0,
//     "three-toed sloth": 500.0,
//     "orangutan": 1000.0,
//     "gorilla": 1500.0,
//     "chimpanzee": 1000.0,
//     "gibbon": 800.0,
//     "siamang": 800.0,
//     "guenon": 300.0,
//     "patas": 400.0,
//     "baboon": 500.0,
//     "macaque": 400.0,
//     "langur": 300.0,
//     "colobus": 400.0,
//     "proboscis monkey": 800.0,
//     "marmoset": 300.0,
//     "capuchin": 400.0,
//     "howler monkey": 600.0,
//     "titi": 400.0,
//     "spider monkey": 500.0,
//     "squirrel monkey": 400.0,
//     "Madagascar cat": 500.0,
//     "indri": 800.0,
//     "Indian elephant": 5000.0,
//     "African elephant": 6000.0,
//     "lesser panda": 1000.0,
//     "giant panda": 2000.0,
//     "barracouta": 20.0,
//     "eel": 10.0,
//     "coho": 20.0,
//     "rock beauty": 20.0,
//     "anemone fish": 10.0,
//     "sturgeon": 50.0,
//     "gar": 30.0,
//     "lionfish": 20.0,
//     "puffer": 30.0,
//     "abacus": 10.0,
//     "abaya": 50.0,
//     "academic gown": 80.0,
//     "accordion": 100.0,
//     "acoustic guitar": 150.0,
//     "aircraft carrier": 5000.0,
//     "airliner": 10000.0,
//     "airship": 8000.0,
//     "altar": 300.0,
//     "ambulance": 3000.0,
//     "amphibian": 500.0,
//     "analog clock": 50.0,
//     "apiary": 200.0,
//     "apron": 30.0,
//     "ashcan": 40.0,
//     "assault rifle": 1500.0,
//     "backpack": 50.0,
//     "bakery": 200.0,
//     "balance beam": 300.0,
//     "balloon": 20.0,
//     "ballpoint": 10.0,
//     "Band Aid": 5.0,
//     "banjo": 150.0,
//     "bannister": 80.0,
//     "barbell": 100.0,
//     "barber chair": 500.0,
//     "barbershop": 300.0,
//     "barn": 2000.0,
//     "barometer": 100.0,
//     "barrel": 200.0,
//     "barrow": 80.0,
//     "baseball": 30.0,
//     "basketball": 40.0,
//     "bassinet": 100.0,
//     "bassoon": 200.0,
//     "bathing cap": 20.0,
//     "bath towel": 30.0,
//     "bathtub": 500.0,
//     "beach wagon": 300.0,
//     "beacon": 200.0,
//     "beaker": 50.0,
//     "bearskin": 500.0,
//     "beer bottle": 5.0,
//     "beer glass": 10.0,
//     "bell cote": 150.0,
//     "bib": 5.0,
//     "bicycle-built-for-two": 100.0,
//     "bikini": 50.0,
//     "binder": 20.0,
//     "binoculars": 50.0,
//     "birdhouse": 100.0,
//     "boathouse": 1000.0,
//     "bobsled": 300.0,
//     "bolo tie": 30.0,
//     "bonnet": 50.0,
//     "bookcase": 300.0,
//     "bookshop": 500.0,
//     "bottlecap": 2.0,
//     "bow": 20.0,
//     "bow tie": 10.0,
//     "brass": 50.0,
//     "brassiere": 30.0,
//     "breakwater": 1000.0,
//     "breastplate": 200.0,
//     "broom": 30.0,
//     "bucket": 20.0,
//     "buckle": 10.0,
//     "bulletproof vest": 500.0,
//     "bullet train": 10000.0,
//     "butcher shop": 1000.0,
//     "cab": 500.0,
//     "caldron": 100.0,
//     "candle": 10.0,
//     "cannon": 1000.0,
//     "canoe": 500.0,
//     "can opener": 10.0,
//     "cardigan": 100.0,
//     "car mirror": 50.0,
//     "carousel": 2000.0,
//     "carpenter's kit": 200.0,
//     "carton": 5.0,
//     "car wheel": 50.0,
//     "cash machine": 2000.0,
//     "cassette": 10.0,
//     "cassette player": 100.0,
//     "castle": 5000.0,
//     "catamaran": 1000.0,
//     "CD player": 150.0,
//     "cello": 500.0,
//     "cellular telephone": 200.0,
//     "chain": 20.0,
//     "chainlink fence": 200.0,
//     "chain mail": 300.0,
//     "chain saw": 200.0,
//     "chest": 300.0,
//     "chiffonier": 200.0,
//     "chime": 50.0,
//     "china cabinet": 500.0,
//     "Christmas stocking": 20.0,
//     "church": 2000.0,
//     "cinema": 2000.0,
//     "cleaver": 50.0,
//     "cliff dwelling": 3000.0,
//     "cloak": 50.0,
//     "clog": 50.0,
//     "cocktail shaker": 50.0,
//     "coffee mug": 10.0,
//     "coffeepot": 30.0,
//     "coil": 20.0,
//     "combination lock": 30.0,
//     "computer keyboard": 50.0,
//     "confectionery": 500.0,
//     "container ship": 10000.0,
//     "convertible": 8000.0,
//     "corkscrew": 10.0,
//     "cornet": 100.0,
//     "cowboy boot": 150.0,
//     "cowboy hat": 100.0,
//     "cradle": 100.0,
//     "crane": 3000.0,
//     "crash helmet": 50.0,
//     "crate": 30.0,
//     "crib": 100.0,
//     "Crock Pot": 100.0,
//     "croquet ball": 20.0,
//     "crutch": 50.0,
//     "cuirass": 300.0,
//     "dam": 5000.0,
//     "desk": 300.0,
//     "desktop computer": 1000.0,
//     "dial telephone": 100.0,
//     "diaper": 10.0,
//     "digital clock": 50.0,
//     "digital watch": 50.0,
//     "dining table": 500.0,
//     "dishrag": 5.0,
//     "dishwasher": 500.0,
//     "disk brake": 50.0,
//     "dock": 2000.0,
//     "dogsled": 1000.0,
//     "dome": 3000.0,
//     "doormat": 20.0,
//     "drilling platform": 50000.0,
//     "drum": 200.0,
//     "drumstick": 50.0,
//     "dumbbell": 50.0,
//     "Dutch oven": 100.0,
//     "electric fan": 50.0,
//     "electric guitar": 300.0,
//     "electric locomotive": 5000.0,
//     "entertainment center": 500.0,
//     "envelope": 5.0,
//     "espresso maker": 50.0,
//     "face powder": 10.0,
//     "feather boa": 50.0,
//     "file": 10.0,
//     "fireboat": 1000.0,
//     "fire engine": 3000.0,
//     "fire screen": 100.0,
//     "flagpole": 200.0,
//     "flute": 100.0,
//     "folding chair": 50.0,
//     "football helmet": 200.0,
//     "forklift": 3000.0,
//     "fountain": 500.0,
//     "fountain pen": 50.0,
//     "four-poster": 1000.0,
//     "freight car": 3000.0,
//     "French horn": 300.0,
//     "frying pan": 30.0,
//     "fur coat": 1000.0,
//     "garbage truck": 3000.0,
//     "gasmask": 100.0,
//     "gas pump": 500.0,
//     "goblet": 10.0,
//     "go-kart": 1000.0,
//     "golf ball": 10.0,
//     "golfcart": 1000.0,
//     "gondola": 2000.0,
//     "gong": 100.0,
//     "gown": 200.0,
//     "grand piano": 5000.0,
//     "greenhouse": 1000.0,
//     "grille": 100.0,
//     "grocery store": 5000.0,
//     "guillotine": 3000.0,
//     "hair slide": 10.0,
//     "hair spray": 10.0,
//     "half track": 5000.0,
//     "hammer": 20.0,
//     "hamper": 30.0,
//     "hand blower": 50.0,
//     "hand-held computer": 500.0,
//     "handkerchief": 5.0,
//     "hard disc": 100.0,
//     "harmonica": 50.0,
//     "harp": 300.0,
//     "harvester": 3000.0,
//     "hatchet": 20.0,
//     "holster": 50.0,
//     "home theater": 500.0,
//     "honeycomb": 5.0,
//     "hook": 10.0,
//     "hoopskirt": 100.0,
//     "horizontal bar": 300.0,
//     "horse cart": 1000.0,
//     "hourglass": 50.0,
//     "iPod": 300.0,
//     "iron": 30.0,
//     "jack-o'-lantern": 20.0,
//     "jean": 50.0,
//     "jeep": 5000.0,
//     "jersey": 100.0,
//     "jigsaw puzzle": 30.0,
//     "jinrikisha": 1000.0,
//     "joystick": 50.0,
//     "kimono": 100.0,
//     "knee pad": 20.0,
//     "knot": 10.0,
//     "lab coat": 100.0,
//     "ladle": 20.0,
//     "lampshade": 30.0,
//     "laptop": 1000.0,
//     "lawn mower": 500.0,
//     "lens cap": 10.0,
//     "letter opener": 10.0,
//     "library": 2000.0,
//     "lifeboat": 1000.0,
//     "lighter": 10.0,
//     "limousine": 8000.0,
//     "liner": 10000.0,
//     "lipstick": 10.0,
//     "Loafer": 50.0,
//     "lotion": 10.0,
//     "loudspeaker": 100.0,
//     "loupe": 30.0,
//     "lumbermill": 3000.0,
//     "magnetic compass": 50.0,
//     "mailbag": 20.0,
//     "mailbox": 50.0,
//     "maillot": 100.0,
//     "manhole cover": 50.0,
//     "maraca": 20.0,
//     "marimba": 200.0,
//     "mask": 50.0,
//     "matchstick": 5.0,
//     "maypole": 300.0,
//     "maze": 1000.0,
//     "measuring cup": 10.0,
//     "medicine chest": 200.0,
//     "megalith": 5000.0,
//     "microphone": 100.0,
//     "microwave": 200.0,
//     "military uniform": 300.0,
//     "milk can": 50.0,
//     "minibus": 3000.0,
//     "miniskirt": 50.0,
//     "minivan": 5000.0,
//     "missile": 100000.0,
//     "mitten": 10.0,
//     "mixing bowl": 20.0,
//     "mobile home": 5000.0,
//     "Model T": 5000.0,
//     "modem": 100.0,
//     "monastery": 2000.0,
//     "monitor": 500.0,
//     "moped": 1000.0,
//     "mortar": 100.0,
//     "mortarboard": 100.0,
//     "mosque": 3000.0,
//     "mosquito net": 50.0,
//     "motor scooter": 1000.0,
//     "mountain bike": 1000.0,
//     "mountain tent": 200.0,
//     "mouse": 20.0,
//     "mousetrap": 20.0,
//     "moving van": 3000.0,
//     "muzzle": 50.0,
//     "nail": 5.0,
//     "neck brace": 30.0,
//     "necklace": 50.0,
//     "nipple": 10.0,
//     "notebook": 50.0,
//     "obelisk": 2000.0,
//     "oboe": 200.0,
//     "ocarina": 50.0,
//     "odometer": 50.0,
//     "oil filter": 10.0,
//     "organ": 500.0,
//     "oscilloscope": 200.0,
//     "overskirt": 50.0,
//     "oxcart": 1000.0,
//     "oxygen mask": 50.0,
//     "packet": 5.0,
//     "paddle": 20.0,
//     "paddlewheel": 2000.0,
//     "padlock": 20.0,
//     "paintbrush": 10.0,
//     "pajama": 50.0,
//     "palace": 5000.0,
//     "panpipe": 100.0,
//     "paper towel": 5.0,
//     "parachute": 1000.0,
//     "parallel bars": 300.0,
//     "park bench": 300.0,
//     "parking meter": 200.0,
//     "passenger car": 5000.0,
//     "patio": 1000.0,
//     "pay-phone": 200.0,
//     "pedestal": 100.0,
//     "pencil box": 20.0,
//     "pencil sharpener": 10.0,
//     "perfume": 30.0,
//     "Petri dish": 10.0,
//     "photocopier": 500.0,
//     "pick": 10.0,
//     "pickelhaube": 50.0,
//     "picket fence": 300.0,
//     "pickup": 5000.0,
//     "pier": 2000.0,
//     "piggy bank": 10.0,
//     "pill bottle": 5.0,
//     "pillow": 30.0,
//     "ping-pong ball": 5.0,
//     "pinwheel": 10.0,
//     "pirate": 3000.0,
//     "pitcher": 20.0,
//     "plane": 10000.0,
//     "planetarium": 2000.0,
//     "plastic bag": 5.0,
//     "plate rack": 30.0,
//     "plow": 1000.0,
//     "plunger": 10.0,
//     "Polaroid camera": 200.0,
//     "pole": 20.0,
//     "police van": 5000.0,
//     "poncho": 50.0,
//     "pool table": 1000.0,
//     "pop bottle": 2.0,
//     "pot": 30.0,
//     "potter's wheel": 100.0,
//     "power drill": 50.0,
//     "prayer rug": 50.0,
//     "printer": 200.0,
//     "prison": 5000.0,
//     "projectile": 500.0,
//     "projector": 200.0,
//     "puck": 10.0,
//     "punching bag": 100.0,
//     "purse": 50.0,
//     "quill": 10.0,
//     "quilt": 50.0,
//     "racer": 1000.0,
//     "racket": 30.0,
//     "radiator": 50.0,
//     "radio": 100.0,
//     "radio telescope": 5000.0,
//     "rain barrel": 50.0,
//     "recreational vehicle": 5000.0,
//     "reel": 50.0,
//     "reflex camera": 200.0,
//     "refrigerator": 1000.0,
//     "remote control": 50.0,
//     "restaurant": 5000.0,
//     "revolver": 300.0,
//     "rifle": 300.0,
//     "rocking chair": 200.0,
//     "rotisserie": 100.0,
//     "rubber eraser": 5.0,
//     "rugby ball": 20.0,
//     "rule": 5.0,
//     "running shoe": 50.0,
//     "safe": 300.0,
//     "safety pin": 5.0,
//     "saltshaker": 10.0,
//     "sandal": 50.0,
//     "sarong": 50.0,
//     "sax": 200.0,
//     "scabbard": 20.0,
//     "scale": 20.0,
//     "school bus": 5000.0,
//     "schooner": 5000.0,
//     "scoreboard": 200.0,
//     "screen": 300.0,
//     "screw": 5.0,
//     "screwdriver": 10.0,
//     "seat belt": 20.0,
//     "sewing machine": 200.0,
//     "shield": 100.0,
//     "shoe shop": 500.0,
//     "shoji": 300.0,
//     "shopping basket": 30.0,
//     "shopping cart": 100.0,
//     "shovel": 20.0,
//     "shower cap": 10.0,
//     "shower curtain": 30.0,
//     "ski": 100.0,
//     "ski mask": 10.0,
//     "sleeping bag": 50.0,
//     "slide rule": 50.0,
//     "sliding door": 200.0,
//     "slot": 100.0,
//     "snorkel": 20.0,
//     "snowmobile": 1000.0,
//     "snowplow": 5000.0,
//     "soap dispenser": 20.0,
//     "soccer ball": 20.0,
//     "sock": 5.0,
//     "solar dish": 1000.0,
//     "sombrero": 50.0,
//     "soup bowl": 10.0,
//     "space bar": 20.0,
//     "space heater": 50.0,
//     "space shuttle": 10000000.0,
//     "spatula": 10.0,
//     "speedboat": 5000.0,
//     "spider web": 5.0,
//     "spindle": 50.0,
//     "sports car": 10000.0,
//     "spotlight": 100.0,
//     "stage": 2000.0,
//     "steam locomotive": 5000.0,
//     "steel arch bridge": 10000.0,
//     "steel drum": 100.0,
//     "stethoscope": 50.0,
//     "stole": 100.0,
//     "stone wall": 3000.0,
//     "stopwatch": 30.0,
//     "stove": 500.0,
//     "strainer": 10.0,
//     "streetcar": 3000.0,
//     "stretcher": 100.0,
//     "studio couch": 500.0,
//     "stupa": 2000.0,
//     "submarine": 5000000.0,
//     "suit": 500.0,
//     "sundial": 200.0,
//     "sunglass": 20.0,
//     "sunglasses": 20.0,
//     "sunscreen": 10.0,
//     "suspension bridge": 10000.0,
//     "swab": 10.0,
//     "sweatshirt": 50.0,
//     "swimming trunks": 50.0,
//     "swing": 100.0,
//     "switch": 10.0,
//     "syringe": 20.0,
//     "table lamp": 50.0,
//     "tank": 1000000.0,
//     "tape player": 100.0,
//     "teapot": 30.0,
//     "teddy": 50.0,
//     "television": 500.0,
//     "tennis ball": 10.0,
//     "thatch": 100.0,
//     "theater curtain": 2000.0,
//     "thimble": 5.0,
//     "thresher": 3000.0,
//     "throne": 500.0,
//     "tile roof": 3000.0,
//     "toaster": 30.0,
//     "tobacco shop": 500.0,
//     "toilet seat": 20.0,
//     "torch": 20.0,
//     "totem pole": 1000.0,
//     "tow truck": 5000.0,
//     "toyshop": 500.0,
//     "tractor": 5000.0,
//     "trailer truck": 10000.0,
//     "tray": 10.0,
//     "trench coat": 100.0,
//     "tricycle": 200.0,
//     "trimaran": 10000.0,
//     "tripod": 50.0,
//     "triumphal arch": 2000.0,
//     "trolleybus": 5000.0,
//     "trombone": 300.0,
//     "tub": 100.0,
//     "turnstile": 100.0,
//     "typewriter keyboard": 50.0,
//     "umbrella": 30.0,
//     "unicycle": 300.0,
//     "upright": 500.0,
//     "vacuum": 100.0,
//     "vase": 30.0,
//     "vault": 5000.0,
//     "velvet": 100.0,
//     "vending machine": 1000.0,
//     "vestment": 100.0,
//     "viaduct": 10000.0,
//     "violin": 300.0,
//     "volleyball": 20.0,
//     "waffle iron": 30.0,
//     "wall clock": 50.0,
//     "wallet": 30.0,
//     "wardrobe": 200.0,
//     "warplane": 10000000.0,
//     "washbasin": 50.0,
//     "washer": 500.0,
//     "water bottle": 10.0,
//     "water jug": 20.0,
//     "water tower": 2000.0,
//     "whiskey jug": 20.0,
//     "whistle": 10.0,
//     "wig": 50.0,
//     "window screen": 50.0,
//     "window shade": 30.0,
//     "Windsor tie": 20.0,
//     "wine bottle": 20.0,
//     "wing": 300.0,
//     "wok": 30.0,
//     "wooden spoon": 10.0,
//     "wool": 100.0,
//     "worm fence": 300.0,
//     "wreck": 5000.0,
//     "yawl": 5000.0,
//     "yurt": 10000.0,
//     "web site": 500.0,
//     "comic book": 10.0,
//     "crossword puzzle": 10.0,
//     "street sign": 50.0,
//     "traffic light": 100.0,
//     "book jacket": 10.0,
//     "menu": 5.0,
//     "plate": 5.0,
//     "guacamole": 5.0,
//     "consomme": 10.0,
//     "hot pot": 20.0,
//     "trifle": 30.0,
//     "ice cream": 10.0,
//     "ice lolly": 5.0,
//     "French loaf": 5.0,
//     "bagel": 5.0,
//     "pretzel": 5.0,
//     "cheeseburger": 10.0,
//     "hotdog": 5.0,
//     "mashed potato": 10.0,
//     "head cabbage": 5.0,
//     "broccoli": 5.0,
//     "cauliflower": 5.0,
//     "zucchini": 5.0,
//     "spaghetti squash": 5.0,
//     "acorn squash": 5.0,
//     "butternut squash": 5.0,
//     "cucumber": 5.0,
//     "artichoke": 10.0,
//     "bell pepper": 5.0,
//     "cardoon": 10.0,
//     "mushroom": 5.0,
//     "Granny Smith": 5.0,
//     "strawberry": 5.0,
//     "orange": 5.0,
//     "lemon": 5.0,
//     "fig": 5.0,
//     "pineapple": 5.0,
//     "banana": 5.0,
//     "jackfruit": 10.0,
//     "custard apple": 10.0,
//     "pomegranate": 5.0,
//     "hay": 5.0,
//     "carbonara": 10.0,
//     "chocolate sauce": 10.0,
//     "dough": 10.0,
//     "meat loaf": 10.0,
//     "pizza": 10.0,
//     "potpie": 10.0,
//     "burrito": 10.0,
//     "red wine": 20.0,
//     "espresso": 5.0,
//     "cup": 5.0,
//     "eggnog": 5.0,
//     "alp": 1000.0,
//     "bubble": 10.0,
//     "cliff": 2000.0,
//     "coral reef": 1000.0,
//     "geyser": 10000.0,
//     "lakeside": 2000.0,
//     "promontory": 3000.0,
//     "sandbar": 500.0,
//     "seashore": 2000.0,
//     "valley": 2000.0,
//     "volcano": 100000.0,
//     "ballplayer": 200.0,
//     "groom": 200.0,
//     "scuba diver": 200.0,
//     "rapeseed": 5.0,
//     "daisy": 5.0,
//     "yellow lady's slipper": 10.0,
//     "corn": 5.0,
//     "acorn": 5.0,
//     "hip": 5.0,
//     "buckeye": 5.0,
//     "coral fungus": 10.0,
//     "agaric": 10.0,
//     "gyromitra": 10.0,
//     "stinkhorn": 10.0,
//     "earthstar": 10.0,
//     "hen-of-the-woods": 10.0,
//     "bolete": 10.0,
//     "ear": 5.0,
//     "toilet tissue": 5.0,
//   };
// }
//---------------------------------------------------------------------------------------------
// const int IMAGE_HEIGHT = 256;
// const int IMAGE_WIDTH = 256;
//
// class KrccsnetEncoder {
//   static Future<int> _encode(Interpreter encoderInterpreter,
//       List<dynamic> input, List<dynamic> output, int times) async {
//     if (times <= 0) {
//       throw Exception("times can't be $times, must > 0");
//     }
//     var start = DateTime.now().millisecondsSinceEpoch;
//     for (int roll = 0; roll < times; ++roll) {
//       encoderInterpreter.run(input, output);
//     }
//     var end = DateTime.now().millisecondsSinceEpoch;
//     int krccsnetTime = (end - start) ~/ times;
//     return krccsnetTime;
//   }
//
//   // static Future<int> _decode(Interpreter decoderInterpreter,
//   //     List<dynamic> input, List<dynamic> output, int times) async {
//   static Future<int> _decode(Interpreter decoderInterpreter,
//       List<dynamic> input, Map<int, Object> output, int times) async {
//     if (times <= 0) {
//       throw Exception("times can't be $times, must > 0");
//     }
//     var start = DateTime.now().millisecondsSinceEpoch;
//     for (int roll = 0; roll < times; ++roll) {
//       decoderInterpreter.runForMultipleInputs([input], output);
//     }
//     var end = DateTime.now().millisecondsSinceEpoch;
//     int krccsnetTime = (end - start) ~/ times;
//     return krccsnetTime;
//   }
//
//   static Future<int> _encodeJpeg(ByteBuffer input, int times) async {
//     // prepare jpeg
//
//     image_utils.JpegEncoder jpegEncoder = image_utils.JpegEncoder(quality: 100);
//     image_utils.Image jpegEncoderTestImage = image_utils.Image.fromBytes(
//         IMAGE_WIDTH, IMAGE_HEIGHT, input.asUint8List());
//
//     var start = DateTime.now().millisecondsSinceEpoch;
//     for (int roll = 0; roll < times; ++roll) {
//       jpegEncoder.encodeImage(jpegEncoderTestImage);
//     }
//     var end = DateTime.now().millisecondsSinceEpoch;
//
//     int jpegTime = (end - start) ~/ times;
//     return jpegTime;
//   }
//
//   static List<dynamic> getEncodeInputTensor(image_utils.Image rawImage) {
//     // var converted = rawImage.
//     // convert 2 y cb cr
//     // rawImage.convert(format: image_utils.FormatType)
//     // 提取y通道
//     // fp32 / int8
//     //
//     var input = List<double>.filled(IMAGE_HEIGHT * IMAGE_WIDTH, 0.0)
//         .reshape([1, 1, IMAGE_HEIGHT, IMAGE_WIDTH]);
//     // fill with lumianceNormalized, shape: n c h w
//     // for (int ch = 0; ch < IMAGE_HEIGHT; ++ch) {
//     //   for (int cw = 0; cw < IMAGE_WIDTH; ++cw) {
//     //     input[0][0][ch][cw] = rawImage.getPixel(cw, ch).luminanceNormalized;
//     //   }
//     // }
//     // fill with lumianceNormalized, shape: n c h w
//     for (int ch = 0; ch < IMAGE_HEIGHT; ++ch) {
//       for (int cw = 0; cw < IMAGE_WIDTH; ++cw) {
//         input[0][0][ch][cw] = rawImage.getPixel(cw, ch);
//       }
//     }
//     return input;
//   }
//
//   static List<dynamic> getEncodeOutputTensor() {
//     // fill output , shape 1, 2, 128, 128
//     var output = List<double>.filled(IMAGE_HEIGHT * IMAGE_WIDTH ~/ 2, 0.0)
//         .reshape([1, 2, IMAGE_HEIGHT ~/ 2, IMAGE_WIDTH ~/ 2]);
//     return output;
//   }
//
//   static List<dynamic> getDecodeInputTensor() {
//     var input = List<double>.filled(IMAGE_HEIGHT * IMAGE_WIDTH ~/ 2, 0.0)
//         .reshape([1, 2, IMAGE_HEIGHT ~/ 2, IMAGE_WIDTH ~/ 2]);
//     return input;
//   }
//
//   static List<dynamic> getDecodeOutputTensor() {
//     var output = List<double>.filled(IMAGE_HEIGHT * IMAGE_WIDTH, 0.0)
//         .reshape([1, 1, IMAGE_HEIGHT, IMAGE_WIDTH]);
//     return output;
//   }
//
//   static void encode(List<Object> options) async {
//     SendPort sendPort = options[0] as SendPort;
//     Interpreter encoderInterpreter = Interpreter.fromAddress(options[2] as int);
//     File(options[1] as String).readAsBytes().then((imageBytes) async {
//       image_utils.Decoder? rawImageDecoder =
//       image_utils.findDecoderForData(imageBytes);
//
//       if (rawImageDecoder == null) {
//         throw Exception("Format not supported.");
//       }
//
//       var rawImage = rawImageDecoder.decodeImage(imageBytes)!;
//       var input = getEncodeInputTensor(rawImage);
//       var output = getEncodeOutputTensor();
//
//       int krccsnetTime = await _encode(encoderInterpreter, input, output, 1);
//       sendPort.send([krccsnetTime, output]);
//     });
//   }
//
//   static void decode(List<Object> options) async {
//     SendPort sendPort = options[0] as SendPort;
//     List<dynamic> input = options[1] as List<dynamic>;
//     Interpreter decoderInterpreter = Interpreter.fromAddress(options[2] as int);
//     // var output = getDecodeOutputTensor();
//     // await _decode(decoderInterpreter, input, output, 1);
//     // List<dynamic> ->
//
//     var output = <int, Object>{};
//     output[0] = getDecodeOutputTensor();
//     output[1] = getDecodeOutputTensor();
//     await _decode(decoderInterpreter, input, output, 1);
//
//     var rawImage = image_utils.Image(256, 256);
//     for (int ch = 0; ch < IMAGE_HEIGHT; ++ch) {
//       for (int cw = 0; cw < IMAGE_WIDTH; ++cw) {
//         double c = (output[1]! as List<dynamic>)[0][0][ch][cw] * 256.0;
//         rawImage.setPixelRgba(cw, ch, c.toInt(), c.toInt(), c.toInt(), 255);
//       }
//     }
//
//     sendPort.send(rawImage.getBytes());
//
//     /// 1. read file(byte stream)
//     /// 2. recover to Tensor
//     /// 3. run
//   }
//
//   static void benchmark(List<Object> options) async {
//     SendPort sendPort = options[0] as SendPort;
//     Interpreter encoderInterpreter = Interpreter.fromAddress(options[2] as int);
//     File(options[1] as String).readAsBytes().then((imageBytes) async {
//       image_utils.Decoder? rawImageDecoder =
//       image_utils.findDecoderForData(imageBytes);
//
//       if (rawImageDecoder == null) {
//         throw Exception("Format not supported.");
//       }
//
//       var rawImage = rawImageDecoder.decodeImage(imageBytes)!;
//
//       var input = getEncodeInputTensor(rawImage);
//       var output = getEncodeOutputTensor();
//
//       var runPass = 10;
//       var krccsnetTime =
//       await _encode(encoderInterpreter, input, output, runPass);
//       var jpegTime = await _encodeJpeg(imageBytes.buffer, runPass);
//       sendPort.send([krccsnetTime, jpegTime]);
//       // piece of shit
//     });
//   }
// }

// class KrccsnetRet {
//   final int time;
//   final
//   const KrccsnetRet({required this.time});
// }





/*
old stuff
  // Padding(
              //   padding: const EdgeInsets.all(8.0),
              //   child: Row(
              //     mainAxisAlignment: MainAxisAlignment.spaceBetween,
              //     children: [
              //       Column(
              //         crossAxisAlignment: CrossAxisAlignment.start,
              //         children: [
              //           // Row(
              //           //   mainAxisAlignment: MainAxisAlignment.spaceAround,
              //           //   children: [
              //           //     // IconButton(
              //           //     //   onPressed: () async {
              //           //     //     cleanResult();
              //           //     //     final result = await imagePicker.pickImage(
              //           //     //       source: ImageSource.camera,
              //           //     //     );
              //           //     //
              //           //     //     imagePath = result?.path;
              //           //     //     setState(() {});
              //           //     //     processImage();
              //           //     //   },
              //           //     //   icon: const Icon(
              //           //     //     Icons.camera,
              //           //     //     size: 64,
              //           //     //   ),
              //           //     // ),
              //           //     // IconButton(
              //           //     //   onPressed: () async {
              //           //     //     cleanResult();
              //           //     //     final result = await imagePicker.pickImage(
              //           //     //       source: ImageSource.gallery,
              //           //     //     );
              //           //     //
              //           //     //     imagePath = result?.path;
              //           //     //     setState(() {});
              //           //     //     processImage();
              //           //     //   },
              //           //     //   icon: const Icon(
              //           //     //     Icons.photo_library_outlined,
              //           //     //     size: 32,
              //           //     //     color: Colors.white,
              //           //     //   ),
              //           //     // ),
              //           //   ],
              //           // ),
              //
              //           // const SizedBox(height: 20),
              //           // const Row(),
              //           // Show model information
              //           // Text(
              //           //   'Input: (shape: ${inputTensor?.shape} type: ${inputTensor?.type})',
              //           // style: TextStyle(color: Colors.green),),
              //           // Text(
              //           //   'Output: (shape: ${outputTensor?.shape} type: ${outputTensor?.type})',
              //           //   style: TextStyle(color: Colors.green),),
              //           // const SizedBox(height: 8),
              //           // // Show picked image information
              //           // if (image != null) ...[
              //           //   Text('Num channels: ${image?.numberOfChannels}',
              //           //     style: TextStyle(color: Colors.green),),
              //           //   // Text(
              //           //   //     'Bits per channel: ${image?}'),
              //           //   // Text('Height: ${image?.height}'),
              //           //   // Text('Width: ${image?.width}'),
              //           // ],
              //         // Divider(),
              //         //   // Show classification result
              //         //   Container(
              //         //     height: 400,
              //         //     width: MediaQuery.of(context).size.width * 0.8,
              //         //     child: Column(
              //         //       crossAxisAlignment: CrossAxisAlignment.start,
              //         //
              //         //       children: [
              //         //
              //         //       ],
              //         //     ),
              //         //   ),
              //
              //
              //         ],
              //       ),
              //     ],
              //   ),
              // ),


 */