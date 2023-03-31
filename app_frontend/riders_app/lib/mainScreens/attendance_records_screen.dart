import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:riders_app/assistantMethods/assistant_methods.dart';
import 'package:riders_app/authentication/auth_screen.dart';
import 'package:riders_app/global/global.dart';
// import 'package:riders_app/global/global.dart';
import 'package:riders_app/widgets/order_card.dart';
import 'package:riders_app/widgets/progress_bar.dart';
import 'package:riders_app/widgets/simple_app_bar.dart';


class AttendanceRecordsScreen extends StatefulWidget {


  @override
  State<AttendanceRecordsScreen> createState() => _AttendanceRecordsScreenState();
}

class _AttendanceRecordsScreenState extends State<AttendanceRecordsScreen> {
  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: SimpleAppBar(title: "Attendance Records",),
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
                        height: 160,
                        width: 160,
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
                "Search Courses",
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
                "Add Class",
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
                "Add Attendance",
                style: TextStyle(
                    color: Colors.black
                ),
              ),
              onTap: () {
                // Navigator.push(context, MaterialPageRoute(builder: (c) => HistoryScreen()));
              },
            ),
            // const Divider(
            //   height: 10,
            //   color: Colors.blueGrey,
            //   thickness: 2,
            // ),
            // ListTile(
            //   leading: const SizedBox(width: 10,),
            //   title: const Text(
            //     "Search",
            //     style: TextStyle(
            //         color: Colors.black
            //     ),
            //   ),
            //   onTap: () {
            //     // Navigator.push(context, MaterialPageRoute(builder: (c) => SearchScreen()));
            //   },
            // ),
            // const Divider(
            //   height: 10,
            //   color: Colors.blueGrey,
            //   thickness: 2,
            // ),
            // ListTile(
            //   leading: const Icon(Icons.add_location, color: Colors.black,),
            //   title: const Text(
            //     "Add New Address",
            //     style: TextStyle(
            //         color: Colors.black
            //     ),
            //   ),
            //   onTap: () {
            //     // Navigator.push(context, MaterialPageRoute(builder: (c) => AddressScreen()));
            //   },
            // ),
            // const Divider(
            //   height: 10,
            //   color: Colors.blueGrey,
            //   thickness: 2,
            // ),
            // ListTile(
            //   leading: const Icon(Icons.exit_to_app, color: Colors.black,),
            //   title: const Text(
            //     "Sign Out",
            //     style: TextStyle(
            //         color: Colors.black
            //     ),
            //   ),
            //   onTap: () {
            //     firebaseAuth.signOut().then((value) {
            //       Navigator.pushReplacement(context, MaterialPageRoute(builder: (c) => const AuthScreen()));
            //     });
            //   },
            // ),
            const Divider(
              height: 10,
              color: Colors.blueGrey,
              thickness: 2,
            ),
          ],
        ),
      ),
    );
  }
}
