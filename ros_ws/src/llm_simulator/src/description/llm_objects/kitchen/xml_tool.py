import xml.etree.ElementTree as ET
import re



with open('sink/sink.xml', 'r') as file:
    xml_data = file.read()

root = ET.fromstring(xml_data)

# Iterate through all 'mesh' elements in 'asset' element
for mesh in root.find('asset').findall('mesh'):
    file_name = mesh.get('file')

    # Find the number at the end of the file name using regex
    match = re.search(r'sink_collision_(\d+)\.obj$', file_name)
    if match:
        number = match.group(1)
        # Set the 'name' attribute
        mesh.set('name', f'sink_collision_{number}')

# Convert the updated XML tree back to a string
updated_xml_data = ET.tostring(root, encoding='unicode')

# Print the updated XML (or you can save it back to a file)
print(updated_xml_data)

# If you want to save the modified XML back to a file, uncomment the following lines
with open('sink/updated_file.xml', 'w') as file:
    file.write(updated_xml_data)


