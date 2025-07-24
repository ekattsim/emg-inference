import asyncio
from bleak import BleakScanner

async def scan_with_callback():
    DEVICE_ADDRESS = "24A528A5-46FC-C425-02D5-E59445D692C3"
    def detection_callback(device, advertisement_data):
        if DEVICE_ADDRESS == advertisement_data.service_uuids:
            print(f"Name: {device.name}, Address: {device.address}")
            print(f"Service UUIDs: {advertisement_data.service_uuids}")
            print(f"Manufacturer Data: {advertisement_data.manufacturer_data}\n")

    scanner = BleakScanner(detection_callback)
    await scanner.start()
    print("starting scan")
    await asyncio.sleep(10.0)  # scan for 5 seconds
    await scanner.stop()

asyncio.run(scan_with_callback())