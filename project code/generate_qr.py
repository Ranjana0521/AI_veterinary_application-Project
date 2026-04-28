import qrcode

animal_id = "101"

qr = qrcode.make(animal_id)
qr.save("animal_101.png")

