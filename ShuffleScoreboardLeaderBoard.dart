import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:shufflebuy/constants.dart';
import 'package:shufflebuy/main.dart';
import 'package:shufflebuy/experiments/RankUpGlow.dart';
import 'package:shufflebuy/model/User.dart';
import 'package:shufflebuy/services/FirebaseHelper.dart';
import 'package:shufflebuy/services/helper.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../profile/shufflebuy_profile.dart';
import 'package:shufflebuy/ui/reusableUtilities/effects/glowing_effect.dart';



// const Color blue = Color(0xff0371dc);
// const Color lovender = Color(0xffaa54ff);
const Color blue = Color(0xff1d07ab);
const Color lovender = Color(0xff0371dc);

class ShuffleScoreboardLeaderBoard extends StatefulWidget {
  late bool useMobileLayout;
  late User currentUser;

  ShuffleScoreboardLeaderBoard(User? currentUser){
    this.currentUser = currentUser!;
  }

  @override
  _LeaderState createState() => _LeaderState();
}

class _LeaderState extends State<ShuffleScoreboardLeaderBoard> {
  late List<SuperCleanHighscoresUser> listCat;
  bool loading = true;
  int total_users = 100;
  //NEW from existing firebase:
  FireStoreUtils _fireStoreUtils = FireStoreUtils();
  //late Future<List<HighscoresModel>> _highscoresFuture;



  @override
  Widget build(BuildContext context) {
    var shortestSide = MediaQuery.of(context).size.shortestSide;

// Determine if we should use mobile layout or not, 600 here is
// a common breakpoint for a typical 7-inch tablet.
    widget.useMobileLayout = shortestSide < 600;

    if (loading) {
      return Container(
          color: Colors.white,
          child: Center(child: CircularProgressIndicator()));
    } else {
      return Scaffold(
          backgroundColor: Colors.indigo[50],
          appBar: PreferredSize(
            preferredSize: Size.fromHeight(100.0),
            child: Container(
              color: Colors.indigo,
              child: SafeArea(
                child: appBar(
                    left: GestureDetector(onTap: () =>  Navigator.of(context).pop(), child: Icon(Icons.arrow_back, color: Colors.white)),
                    title: 'Shufflescore'.tr(),
                    right: GestureDetector(
                        onTap: (){
                          showAlertDialogShuffleScoreHelp(context, "" ,"");
                        },
                        child: Icon(FontAwesomeIcons.question, color: Colors.red))
                ),
              ),
            ),
          ),
          // backgroundColor: Color(0xfff4f4f4),
          body: RefreshIndicator(
            onRefresh: () async {callRefreshChats();},
            child: SingleChildScrollView(
                child: Stack(
              children: <Widget>[
                profile(),
                get_rank(),
                //add a widget to show the total number of users:
                MyAppState.currentUser!.isAdmin == true ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(18.0),
                    child: Container(
                      margin: EdgeInsets.only(top: 20),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          GlowingAuraSquareTransparentWithOpacity(
                            color: Colors.white,
                            opacity: 0.4,
                            child: Text(
                              total_users.toString(),
                              style: TextStyle(
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white),
                            ),
                          ),
                          Text(
                              " Shufflers",
                              style: TextStyle(
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white),
                            ),

                        ],
                      ),
                      ),
                  ),
                )
                :
                    Container(),
                Center(child: powerLevelColorWidget(widget.currentUser)),
              ],
            )),
          ));
    }
  }
  Future<void> callRefreshChats() async {
    setState(() {
      //refresh
      getLeader();
      getTotalUsers();
    });
  }
  Widget profile() {
    return Stack(
      children: <Widget>[
        Image.asset("assets/old_sb_images/back_splash.jpg"),
        // Container(
        //   height: MediaQuery.of(context).size.height * 2,
        //  //grey color HEX 16161d
        //   color: Colors.blue[700],
        // ),
        Center(
            child: Container(
                margin:
                    EdgeInsets.only(top: widget.useMobileLayout ? 150 : 250),
                child: Card(
                    child: Padding(
                        padding: const EdgeInsets.symmetric(
                            vertical: 10, horizontal: 10),
                        child: Column(
                          children: <Widget>[
                            checkUserForVipStatus(listCat[0].user_object_from_original) ?

                            GlowingAura(
                              opacity: 0.75,
                              color: powerLevelColorGenerator(listCat[0].user_object_from_original)!,
                              child: CircleAvatar(
                              radius: 40,
                              backgroundColor: Colors.black26,
                              child: CircleAvatar(
                                  radius: 39,
                                  backgroundColor: Colors.white,
                                  child:  GestureDetector(
                                    onTap: () async{
                                      push(context, UserProfilePage.HighScoresLaunch(
                                          listCat[0].user_object_from_original,
                                          true
                                      ));
                                    },
                                    child: CircleAvatar(
                                      radius: 39,
                                      backgroundColor: Color(0xff571855),
                                      backgroundImage:
                                      listCat[0].imageUrl.length > 5 ? NetworkImage( listCat[0].imageUrl)
                                          : AssetImage('assets/images/circle-outline.gif') as ImageProvider,
                                    ),
                                  ))
                        ),
                            )

                            :

                            CircleAvatar(
                                radius: 40,
                                backgroundColor: Colors.black26,
                                child: CircleAvatar(
                                    radius: 39,
                                    backgroundColor: Colors.white,
                                    child:  GestureDetector(
                                      onTap: () async{
                                        push(context, UserProfilePage.HighScoresLaunch(
                                            listCat[0].user_object_from_original,
                                            true
                                        ));
                                      },
                                      child: CircleAvatar(
                                        radius: 39,
                                        backgroundColor: Color(0xff571855),
                                        backgroundImage:
                                        listCat[0].imageUrl.length > 5 ? NetworkImage( listCat[0].imageUrl)
                                            : AssetImage('assets/images/circle-outline.gif') as ImageProvider,
                                      ),
                                    ))
                            ),
                            Padding(
                              padding: const EdgeInsets.only(top: 20.0),
                              child: Text(
                                listCat[0].username,
                                style: TextStyle(color: blue),
                              ),
                            ),
                            Text(
                              listCat[0].scoreRoundedUpAndDividedBySuperCleanModel(),
                              style: TextStyle(color: blue),
                            )
                          ],
                        ))))),
        Center(
            child: Container(
                margin: EdgeInsets.only(
                    top: widget.useMobileLayout ? 165 : 265,
                    right: (MediaQuery.of(context).size.width / 2) + 50),
                child: Stack(
                  children: <Widget>[
                    Container(
                      margin: EdgeInsets.only(top: 10),
                        child: Card(
                            child: Padding(
                                padding: const EdgeInsets.symmetric(
                                    vertical: 10, horizontal: 10),
                                child: Column(
                                  children: <Widget>[

                                    checkUserForVipStatus(listCat[1].user_object_from_original) ?

                                    GlowingAura(
                                      opacity: 0.75,
                                      color: powerLevelColorGenerator(listCat[1].user_object_from_original)!,
                                      child: CircleAvatar(
                                          radius: 40,
                                          backgroundColor: Colors.black26,
                                          child: CircleAvatar(
                                              radius: 39,
                                              backgroundColor: Colors.white,
                                              child:  GestureDetector(
                                                onTap: () async{
                                                  push(context, UserProfilePage.HighScoresLaunch(
                                                      listCat[1].user_object_from_original,
                                                      true
                                                  ));
                                                },
                                                child: CircleAvatar(
                                                  radius: 39,
                                                  backgroundColor:
                                                  Color(0xff571855),
                                                  backgroundImage:
                                                  listCat[1].imageUrl.length > 5 ? NetworkImage( listCat[1].imageUrl)
                                                      : AssetImage('assets/images/circle-outline.gif') as ImageProvider,
                                                ),
                                              ))),

                                    )

                                        :



                                    CircleAvatar(
                                        radius: 40,
                                        backgroundColor: Colors.black26,
                                        child: CircleAvatar(
                                            radius: 39,
                                            backgroundColor: Colors.white,
                                            child: GestureDetector(
                                              onTap: () async{
                                                push(context, UserProfilePage.HighScoresLaunch(
                                                    listCat[1].user_object_from_original,
                                                    true
                                                ));
                                              },
                                              child: CircleAvatar(
                                                radius: 39,
                                                backgroundColor:
                                                    Color(0xff571855),
                                                backgroundImage:
                                                listCat[1].imageUrl.length > 5 ? NetworkImage( listCat[1].imageUrl)
                                                    : AssetImage('assets/images/circle-outline.gif') as ImageProvider,
                                              ),
                                            ))),


                                    Padding(
                                      padding: const EdgeInsets.only(top: 15.0),
                                      child: Text(
                                        listCat[1].username,
                                        style: TextStyle(color: blue),
                                      ),
                                    ),
                                    Text(
                                      listCat[1].scoreRoundedUpAndDividedBySuperCleanModel(),
                                      style: TextStyle(color: blue),
                                    )
                                  ],
                                )))),
                    Container(
                        margin: EdgeInsets.only(left: 70),
                        child: Image.asset(
                          "assets/old_sb_images/rank2.png",
                          height: 30,
                          width: 30,
                        )),
                  ],
                ))),
        Center(
            child: Container(
                margin: EdgeInsets.only(
                    top: widget.useMobileLayout ? 170 : 270,
                    left: (MediaQuery.of(context).size.width / 2) + 50),
                child: Stack(
                  children: <Widget>[
                    Container(
                      margin: EdgeInsets.only(top: 10),
                      child: Card(
                          child: Padding(
                              padding: const EdgeInsets.symmetric(
                                  vertical: 10, horizontal: 10),
                              child: Column(
                                children: <Widget>[
                                  checkUserForVipStatus(listCat[2].user_object_from_original) ?

                                  GlowingAura(
                                    opacity: 0.75,
                                    color: powerLevelColorGenerator(listCat[2].user_object_from_original)!,
                                    child: CircleAvatar(
                                        radius: 40,
                                        backgroundColor: Colors.black26,
                                        child: CircleAvatar(
                                            radius: 39,
                                            backgroundColor: Colors.white,
                                            child:  GestureDetector(
                                              onTap: () async{
                                                push(context, UserProfilePage.HighScoresLaunch(
                                                    listCat[2].user_object_from_original,
                                                    true
                                                ));
                                              },
                                              child: CircleAvatar(
                                                radius: 39,
                                                backgroundColor: Color(0xff571855),
                                                backgroundImage:
                                                listCat[2].imageUrl.length > 5 ? NetworkImage( listCat[2].imageUrl)
                                                    : AssetImage('assets/images/circle-outline.gif') as ImageProvider,
                                              ),
                                            ))),
                                  )

                                      :



                                  CircleAvatar(
                                      radius: 40,
                                      backgroundColor: Colors.black26,
                                      child: CircleAvatar(
                                          radius: 39,
                                          backgroundColor: Colors.white,
                                          child: GestureDetector(
                                            onTap: () async{
                                              push(context, UserProfilePage.HighScoresLaunch(
                                                  listCat[2].user_object_from_original,
                                                  true
                                              ));
                                            },
                                            child: CircleAvatar(
                                              radius: 39,
                                              backgroundColor: Color(0xff571855),
                                              backgroundImage:
                                              listCat[2].imageUrl.length > 5 ? NetworkImage( listCat[2].imageUrl)
                                                  : AssetImage('assets/images/circle-outline.gif') as ImageProvider,
                                            ),
                                          ))),
                                  Padding(
                                    padding: const EdgeInsets.only(top: 10.0),
                                    child: Text(
                                      listCat[2].username,
                                      style: TextStyle(color: blue),
                                    ),
                                  ),
                                  Text(
                                    listCat[2].scoreRoundedUpAndDividedBySuperCleanModel(),
                                    style: TextStyle(color: blue),
                                  )
                                ],
                              ))),
                    ),
                    Container(
                        margin: EdgeInsets.only(
                          left: 70,
                        ),
                        child: Image.asset(
                          "assets/old_sb_images/rank3.png",
                          height: 30,
                          width: 30,
                        )),
                  ],
                ))),
        // Container(
        //   margin: EdgeInsets.only(top: widget.useMobileLayout ? 320 : 420),
        //   height: 0,
        //   decoration: BoxDecoration(
        //     // Box decoration takes a gradient
        //     gradient: LinearGradient(
        //       // Where the linear gradient begins and ends
        //       begin: Alignment.topLeft,
        //       end: Alignment.bottomRight,
        //       // Add one stop for each color. Stops should increase from 0 to 1
        //       stops: [0.3, 0.7],
        //       colors: [blue, lovender],
        //     ),
        //   ),
        // ),
        Container(
            margin: EdgeInsets.only(top: widget.useMobileLayout ? 330 : 430),
            decoration: BoxDecoration(
              // Box decoration takes a gradient
              gradient: LinearGradient(
                // Where the linear gradient begins and ends
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                // Add one stop for each color. Stops should increase from 0 to 1
                stops: [0.3, 0.7],
                colors: [blue, lovender],
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: <Widget>[
                  Text("Rank", style: TextStyle(color: Colors.white)),
                  Text("User", style: TextStyle(color: Colors.white)),
                  Text("Score", style: TextStyle(color: Colors.white)),
                ],
              ),
            )),
        Container(
            margin: EdgeInsets.only(
                top: widget.useMobileLayout ? 140 : 240,
                left: (MediaQuery.of(context).size.width) / 2 + 35),
            child: Image.asset(
              "assets/old_sb_images/rank1.png",
              height: 30,
              width: 30,
            )),

      ],
    );
  }

  Widget get_rank() {
    return MediaQuery.removePadding(
        removeTop: true,
        context: context,
        child: Padding(
          padding: EdgeInsets.only(top: widget.useMobileLayout ? 370 : 470),
          child: ListView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: listCat.length,
              itemBuilder: (BuildContext context, int index) {
                return GestureDetector(
                  onTap: () async{
                    push(context, UserProfilePage.HighScoresLaunch(
                        listCat[index].user_object_from_original,
                        true
                    ));
                  },
                  child: Center(
                      child: listCat == null
                          ? CircularProgressIndicator()
                          : (listCat.isEmpty
                              ? Center(
                                  child: Text(
                                    "No Players Found...",
                                    style: TextStyle(
                                        color: Colors.black,
                                        fontWeight: FontWeight.bold,
                                        fontSize: 20),
                                  ),
                                )
                              : Container(
                                  margin: EdgeInsets.symmetric(horizontal: 10),
                                  child: Card(
                                    color: Colors.white,
                                    child: Padding(
                                      padding: const EdgeInsets.symmetric(
                                          vertical: 8, horizontal: 15),
                                      child: Row(
                                        children: <Widget>[
                                          Container(
                                            child: Text(

                                                (index+1)
                                                        .toString() +
                                                    ".",
                                                style: TextStyle(
                                                    color: Color(0xff4a81c0))),
                                          ),
                                          Padding(
                                            padding:
                                                const EdgeInsets.only(left: 10.0),
                                            child:

                                            checkUserForVipStatus(listCat[index].user_object_from_original)?
                                            GlowingAura(
                                              opacity: 0.75,
                                              color: powerLevelColorGenerator(listCat[index].user_object_from_original)!,
                                              child: CircleAvatar(
                                                backgroundColor: Color(0xff4a81c0),
                                                radius: 20,
                                                backgroundImage:
                                                listCat[index].imageUrl.length > 5 ? NetworkImage( listCat[index].imageUrl)
                                                    : AssetImage('assets/images/circle-outline.gif') as ImageProvider,



                                              ),
                                            )
                                                :
                                            CircleAvatar(
                                              backgroundColor: Color(0xff4a81c0),
                                              radius: 20,
                                              backgroundImage:
                                              listCat[index].imageUrl.length > 5 ? NetworkImage( listCat[index].imageUrl)
                                                  : AssetImage('assets/images/circle-outline.gif') as ImageProvider,



                                            )
                                          ),
                                          Expanded(
                                              child: Align(
                                                alignment: Alignment.center,
                                                child: Text(
                                                  listCat[index].username,
                                                  style: TextStyle(
                                                    color: Color(0xff4a81c0),
                                                  ),
                                                ),
                                              )),
                                          // Image.asset(
                                          //   "assets/old_sb_images/smallstar.png",
                                          //   height: 20,
                                          // ),
                                          Padding(
                                            padding:
                                                const EdgeInsets.only(left: 8.0),
                                            child: Text(listCat[index].scoreRoundedUpAndDividedBySuperCleanModel(),
                                                style: TextStyle(
                                                    color: Color(0xff4a81c0))),
                                          )
                                        ],
                                      ),
                                    ),
                                  )))),
                );
              }),
        ));
  }

  @override
  void initState() {
      getLeader();
      getTotalUsers();
    super.initState();


  }


  //helper method to show alert dialog
  showAlertDialogShuffleScoreHelp(BuildContext context, String title, String content) {
    // set up the AlertDialog
    Widget okButton = TextButton(
      child: Text('OK'.tr()),
      onPressed: () {
        Navigator.pop(context);
      },
    );
    CupertinoAlertDialog alert = CupertinoAlertDialog(
      title: Center(child:  Text("My Score: ${MyAppState.currentUser!.scoreRoundedUpAndDividedBy()}")),
      content: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(child: Image.asset('assets/images/premium_account_3.gif', height: 150,width: 150,)),
          SizedBox(height: 15,),
          Center(child: Text("Your ShuffleScore increases by:")),
          SizedBox(height: 15,),
          Text("Selling your items through Shufflebuy"),
          Text("Creating a listing"),
          Text("Marking a listing as sold"),
          Text("Deleting a listing"),
          Text("Buyer confirms purchase"),
        ],
      ),
      actions: [okButton],
    );

    // show the dialog
    showCupertinoDialog(
      context: context,
      builder: (BuildContext context) {
        return alert;
      },
    );

  }

  Future getTotalUsers() async {

    List<SuperCleanHighscoresUser> users = [];
    var users_ref = await _fireStoreUtils.getUserResultList();


    //print(users.length);
    setState(() {
      if(users_ref != null) {
        total_users = users_ref.length;
      }
    });
    return users;
  }
  Future getLeader() async {

    List<SuperCleanHighscoresUser> users = [];
    List _futureScores = await _fireStoreUtils.getHighscoresSimple();

    for (var u in _futureScores) {
      SuperCleanHighscoresUser user = SuperCleanHighscoresUser(u.id, u.score, u.username,  u.image_url, u.user_as_a_real_user_object);
      //print('Adding user: ${user.id} - ${user.imageUrl} - ${user.score} - ${user.username}}');
      users.add(user);
    }

    //sort based on score
    users.sort((b,a) => a.score_rank.compareTo(b.score_rank));


    // for (var list_rank in users) {
    //   print("${list_rank.username} - ${list_rank.score}");
    // }




    //print(users.length);
    setState(() {
      loading = false;
      listCat = users;
    });
    return users;
  }



//  Future<String> getLeader() async {
//    var data = {'access_key': "90336", 'get_leaderboard_detail': "1"};
//    var response = await http.post(base_url, body: data);
//
//
//    var getdata = json.decode(response.body);
//    String total = getdata["total"];
//    if (int.parse(total) > 0) {
//      setState(() {
//        loading = false;
//        listCat = getdata["rows"];
//      });
//    }
//  }



}

//---------------------------------
//Object for json data
//---------------------------------
class SuperCleanHighscoresUser {
  late String id;
  late String imageUrl;
  late String username;
  late String score;
  late int score_rank;
  late User user_object_from_original;

  SuperCleanHighscoresUser(String id, int score, String username, String imageUrl, User user_object_from_original){
    this.id = id;
    this.score = score.toString();
    this.username = username;
    this.imageUrl = imageUrl;
    this.score_rank = score;
    this.user_object_from_original = user_object_from_original;
  }
  String scoreRoundedUpAndDividedBySuperCleanModel(){
    int score_returned = 1;
    //test if score can be divided by 100
    if(this.score_rank % SCORE_DIVISION == 0){
      score_returned = this.score_rank ~/ SCORE_DIVISION;
    }else{
      score_returned = (this.score_rank ~/ SCORE_DIVISION) + 1;
    }
    //round score up to the next whole number
    if(score_returned < 1){
      score_returned = 1;
    }
    debugPrint("score returned: $score_returned");
    return score_returned.toString();
  }
}



class NetworkImageHandlerViewer extends StatefulWidget {
  final String imageURL;
  bool errorFoundInImageLoad = false;
  NetworkImageHandlerViewer({
    Key? key,
    required this.imageURL,
  }) : super(key: key);

  @override
  _NetworkImageHandlerViewerState createState() =>
      _NetworkImageHandlerViewerState();
}

class _NetworkImageHandlerViewerState extends State<NetworkImageHandlerViewer> {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          height: 200,
          // height: MediaQuery.of(context).size.height,
          width: MediaQuery.of(context).size.width,
          // color: Colors.black,
          child: (widget.errorFoundInImageLoad)
              ? Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                "Unable To Fetch Image",
              ),
              IconButton(
                iconSize: 40,
                onPressed: () {
                  setState(() {
                    if (mounted) {
                      print("reloading image again");
                      setState(() {
                        widget.errorFoundInImageLoad = false;
                      });
                    }
                  });
                },
                icon: Icon(Icons.refresh),
              ),
              Text(
                "Tap Refresh Icon To Reload!!!",
              ),
            ],
          )
              : Image.network(
            // widget.question.fileInfo[0].remoteURL,
            widget.imageURL,
            //
            loadingBuilder: (context, child, loadingProgress) =>
            (loadingProgress == null)
                ? child
                : Center(
              child: CircularProgressIndicator(),
            ),
            errorBuilder: (context, error, stackTrace) {
              Future.delayed(
                Duration(milliseconds: 0),
                    () {
                  if (mounted) {
                    setState(() {
                      widget.errorFoundInImageLoad = true;
                    });
                  }
                },
              );
              return SizedBox.shrink();
            },
          ),
        ),
        SizedBox(height: 25),
      ],
    );
  }


}

