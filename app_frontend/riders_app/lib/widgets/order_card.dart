import 'package:cloud_firestore/cloud_firestore.dart';
// import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:riders_app/mainScreens/order_details_screen.dart';
import 'package:riders_app/mainScreens/parcel_delivering_screen.dart';
import 'package:riders_app/mainScreens/parcel_picking_screen.dart';
import 'package:riders_app/models/items.dart';


class OrderCard extends StatelessWidget {

  final int? itemCount;
  final List<DocumentSnapshot>? data;
  final String? orderID;
  final List<String>? separateQuantitiesList;

  OrderCard({
    this.itemCount,
    this.data,
    this.orderID,
    this.separateQuantitiesList,
  });

  String orderStatus = "";
  String orderByUser = "";
  String sellerId = "";
  String addressID = "";
  String purchaserAddress = "";
  double purchaserLat = 0.0;
  double purchaserLng = 0.0;


  getOrderInfo(){
    FirebaseFirestore.instance
        .collection("orders")
        .doc(orderID)
        .get().then((DocumentSnapshot)
    {
      orderStatus = DocumentSnapshot.data()!["status"].toString();
      orderByUser = DocumentSnapshot.data()!["orderBy"].toString();
      sellerId = DocumentSnapshot.data()!["sellerUID"].toString();
      addressID = DocumentSnapshot.data()!["addressID"].toString();
    });
  }

  getPurchaserInfo(){
    FirebaseFirestore.instance
        .collection("users")
        .doc(orderByUser)
        .collection("userAddress")
        .doc(addressID)
        .get().then((snap) {
          purchaserAddress = snap.data()!["fullAddress"].toString();
          purchaserLat = double.parse(snap.data()!["lat"]);
          purchaserLng = double.parse(snap.data()!["lng"]);
    });
  }

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () {
        getOrderInfo();
        if(orderStatus == "normal"|| orderStatus == "ended" ){
          Navigator.push(context, MaterialPageRoute(builder: (c) => OrderDetailsScreen(orderID: orderID)));
        }
        if(orderStatus == "picking"){
          getPurchaserInfo();
          Navigator.push(context, MaterialPageRoute(builder: (c) => ParcelPickingScreen(
            purchaserId: orderByUser,
            purchaserAddress: purchaserAddress,
            purchaserLat: purchaserLat,
            purchaserLng: purchaserLng,
            sellerId: sellerId,
            getOrderID: orderID,
          )));
        }
        if(orderStatus == "delivering"){
          getPurchaserInfo();
          Navigator.push(context, MaterialPageRoute(builder: (c) => ParcelDeliveringScreen(
            purchaserId: orderByUser,
            purchaserAddress: purchaserAddress,
            purchaserLat: purchaserLat,
            purchaserLng: purchaserLng,
            sellerId: sellerId,
            getOrderId: orderID,
          )));
        }


      },
      child: Container(
        decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [
                Colors.black12,
                Colors.white54,
              ],
              begin: FractionalOffset(0.0, 0.0),
              end: FractionalOffset(1.0, 0.0),
              stops: [0.0, 1.0],
              tileMode: TileMode.clamp,
            )
        ),
        padding: const EdgeInsets.all(10),
        margin: const EdgeInsets.all(10),
        height: itemCount! * 125,
        child: ListView.builder(
          itemCount: itemCount,
          physics: const NeverScrollableScrollPhysics(),
          itemBuilder: (context, index){
            Items model = Items.fromJson(data![index].data()! as Map<String, dynamic>);
            return placedOrderDesignWidget(model, context, separateQuantitiesList![index]);
          },
        ),
      ));
  }
}


Widget placedOrderDesignWidget(Items model, BuildContext context, separateQuantitiesList){
  return Container(
    width: MediaQuery.of(context).size.width,
    height: 120,
    color: Colors.grey[200],
    child: Row(
      children: [
        Image.network(model.thumbnailUrl!, width: 120,),
        const SizedBox(width: 10,),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 20,),
              Row(
                mainAxisSize: MainAxisSize.max,
                children: [
                  Expanded(
                    child: Text(
                      model.title!,
                      style: const TextStyle(
                        color: Colors.black,
                        fontSize: 16,
                        fontFamily: "Acme",
                      ),
                    ),
                  ),
                  const SizedBox(width: 10,),
                  const Text(
                    "₹ ",
                    style: TextStyle(
                      fontSize: 16.0,
                      color: Colors.blue,
                    ),
                  ),
                  Text(
                    model.price.toString(),
                    style: const TextStyle(
                      fontSize: 18.0,
                      color: Colors.blue,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10,),
              Row(
                children: [
                  const Text(
                    "x ",
                    style: TextStyle(
                      fontSize: 14.0,
                      color: Colors.black54,
                    ),
                  ),
                  Text(
                    separateQuantitiesList,
                    style: const TextStyle(
                      fontFamily: "Acme",
                      fontSize: 30.0,
                      color: Colors.black54,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    ),
  );
}