import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:riders_app/assistantMethods/assistant_methods.dart';
import 'package:riders_app/authentication/auth_screen.dart';
import 'package:riders_app/global/global.dart';
// import 'package:riders_app/global/global.dart';
import 'package:riders_app/widgets/order_card.dart';
import 'package:riders_app/widgets/progress_bar.dart';
import 'package:riders_app/widgets/simple_app_bar.dart';


class UserDetailsScreen extends StatefulWidget {


  @override
  State<UserDetailsScreen> createState() => _UserDetailsScreenState();
}

class _UserDetailsScreenState extends State<UserDetailsScreen> {

  QuerySnapshot? allUsers;

  @override
  void initState() {

    super.initState();

    FirebaseFirestore.instance
        .collection("students")
        // .where("status", isEqualTo: "approved")
        .get().then((A)
    {
      setState(() {
        allUsers = A;
        print(allUsers!.docs.length);
      });

    });
  }

  @override
  Widget build(BuildContext context) {

    Widget displayVerifiedUsersDesign(){
      if(allUsers != null){
        return ListView.builder(
            padding: const EdgeInsets.all(10),
            itemCount: allUsers!.docs.length,
            itemBuilder: (context, i)
            {
              return Card(
                elevation: 10,
                child: Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: ListTile(
                        leading: Container(
                          width: 65,
                          height: 65,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            image: DecorationImage(
                              image: NetworkImage(allUsers!.docs[i].get("studentAvatarUrl")),
                              fit: BoxFit.fill,
                            ),
                          ),
                        ),
                        // title: Text(
                        //   allUsers!.docs[i].get("studentName"),
                        // ),
                        trailing: Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Column(
                            children: [
                              Text(
                                allUsers!.docs[i].get("studentName"),
                              ),
                              Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  // const Icon(Icons.email, color: Colors.black,),
                                  const SizedBox(width: 20,),
                                  Text(
                                    allUsers!.docs[i].get("studentEmail"),
                                    style: const TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    // Padding(
                    //   padding: const EdgeInsets.all(15.0),
                    //   child: ElevatedButton.icon(
                    //     style: ElevatedButton.styleFrom(
                    //       primary: Colors.red,
                    //
                    //     ),
                    //     onPressed: () {
                    //       // displayDialogBoxForBlockingAccount(allUsers!.docs[i].id);
                    //     },
                    //     icon: const Icon(Icons.person_pin_sharp, color: Colors.white,),
                    //     label: Text(
                    //       "Block this Account".toUpperCase(),
                    //       style: const TextStyle(
                    //         fontSize: 15,
                    //         color: Colors.white,
                    //         letterSpacing: 3,
                    //       ),
                    //     ),
                    //   ),
                    // ),
                  ],
                ),
              );
            }
        );
      }
      else{
        return const Center(
          child: Text(
            "No Record Found",
            style: TextStyle(fontSize: 30,),
          ),
        );
      }

    }

    return SafeArea(
      child: Scaffold(
        appBar: SimpleAppBar(title: "User Details",),
        body: ListView(
          children: [
            //header drawer
            Container(
              padding: const EdgeInsets.only(top: 25, bottom: 10),
              child: Column(
                children: [

                  Material(
                    borderRadius: const BorderRadius.all(Radius.circular(80)),
                    elevation: 10,
                    child: Padding(
                      padding: const EdgeInsets.all(1.0),
                      child: Container(
                        height: 130,
                        width: 130,
                        child: CircleAvatar(
                          backgroundImage: NetworkImage(
                              sharedPreferences!.getString("photoUrl")!
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10,),
                  Text(
                    "Admin",
                    style: const TextStyle(
                        color: Colors.black,
                        fontSize: 20,
                        fontFamily: "Train"
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12,),
            Container(
              padding: const EdgeInsets.only(top: 1.0),
              child: Column(
                children: const [
                  Divider(
                    height: 10,
                    color: Colors.blueGrey,
                    thickness: 2,
                  ),
                ],
              ),
            ),
            ListTile(
              leading: SizedBox(width: 10,),
              title: const Text(
                "Search User",
                style: TextStyle(
                    color: Colors.black
                ),
              ),
              onTap: () {
                // Navigator.pushReplacement(context, MaterialPageRoute(builder: (c) => const HomeScreen()));
              },
            ),
            const Divider(
              height: 10,
              color: Colors.blueGrey,
              thickness: 2,
            ),
            ListTile(
              leading: const SizedBox(width: 10,),
              title: const Text(
                "Add User",
                style: TextStyle(
                    color: Colors.black
                ),
              ),
              onTap: () {
                // Navigator.push(context, MaterialPageRoute(builder: (c) => MyOrdersScreen()));
              },
            ),
            const Divider(
              height: 10,
              color: Colors.blueGrey,
              thickness: 2,
            ),
            ListTile(
              leading: const SizedBox(width: 10,),
              title: const Text(
                "Remove User",
                style: TextStyle(
                    color: Colors.black
                ),
              ),
              onTap: () {
                // Navigator.push(context, MaterialPageRoute(builder: (c) => HistoryScreen()));
              },
            ),

            const Divider(
              height: 10,
              color: Colors.blueGrey,
              thickness: 2,
            ),
            Container(
              width: MediaQuery.of(context).size.width * 0.5,
              height: 400,
              child: displayVerifiedUsersDesign(),
            ),
            // displayVerifiedUsersDesign(),
          ],
        ),
      ),
    );
  }
}



// displayDialogBoxForBlockingAccount(userDocumentID){
//   return showDialog(
//       context: context,
//       builder: (BuildContext context){
//         return AlertDialog(
//           title: const Text(
//             "Block Account",
//             style: TextStyle(
//               fontSize: 25,
//               letterSpacing: 2,
//               fontWeight: FontWeight.bold,
//             ),
//           ),
//           content: const Text(
//             "Do you want to block this account?",
//             style: TextStyle(
//               fontSize: 16,
//               letterSpacing: 2,
//
//             ),
//           ),
//           actions: [
//             ElevatedButton(
//               onPressed: () {
//                 Navigator.pop(context);
//               },
//               child: const Text("No"),
//             ),
//             ElevatedButton(
//               onPressed: () {
//                 Map<String, dynamic> userDataMap = {
//                   "status": "not approved",
//                 };
//
//                 FirebaseFirestore.instance
//                     .collection("users")
//                     .doc(userDocumentID)
//                     .update(userDataMap).then((value)
//                 {
//                   Navigator.push(context, MaterialPageRoute(builder: (c) => const HomeScreen()));
//
//                   SnackBar snackBar = const SnackBar(content: Text(
//                     "Blocked Successfully.",
//                     style: TextStyle(
//                       fontSize: 30,
//                       color: Colors.black,
//                     ),
//                   ),
//                     backgroundColor: Colors.cyan,
//                     duration: Duration(seconds: 2),
//
//                   );
//                   ScaffoldMessenger.of(context).showSnackBar(snackBar);
//                 });
//               },
//               child: const Text("Yes"),
//             ),
//           ],
//         );
//       }
//   );
// }








