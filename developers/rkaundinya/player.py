from enum import Enum

#Hard coded and copied from generatorTradeAndNonTrade.py for now 
#(obv would be good to clean this up)
itemList = ["GOLD", "STONES", "WOOD", "WHEAT", "COAL", "IRON", "ALUMINUM", "HORSES"]
Resources = Enum('Resources', itemList)

#Note - probably a bad way of doing this but just quick prototyping
class ResourceStats:
    def __init__(self, count, value):
        self.count = count
        self.value = value

    def AddToCount(self, count):
        self.count += count

        if self.count < 0:
            self.count = 0

    def GetCount(self):
        return self.count

    def IsEmpty(self):
        return self.count <= 0

    def GetValue(self):
        return self.value

    def __str__(self):
        return "\tCount: " + str(self.count) + "\n\t" + "Value: " + str(self.value)

    

class Player:
    def __init__(self):
        #Dict of resource enum to (val,count) pair
        self.resources = {}
        #Set of resource names for quick check
        self.resourceNames = {}
        #Map to access values of resources
        self.resourceValMap = {}

    #Resources expected to have key=resource enum name, val=amt
    def SetResources(self, resourcesDict):
        self.resources = resourcesDict
        self.resourceNames = {resource.name.lower() : resource for resource in self.resources.keys()}

        for resource in self.resources.keys():
            val = self.resources[resource].GetValue()
            self.resourceValMap[resource] = val

    def HasResource(self, resource):
        if resource in self.resourceNames:
            return True, self.resources[self.resourceNames[resource]]

        return False, -1

    def SetResourceValueMap(self, resource, value):
        self.resourceValMap[resource] = value

    def DebugPrintResources(self):
        for resource in self.resources.keys():
            print("Resource: " + resource.name)
            print(self.resources[resource])

    def DebugPrintResourceValueMap(self):
        for resource in self.resourceValMap.keys():
            print("Resource: " + resource.name, end=", ")
            print("Value: " + str(self.resourceValMap[resource]))
    

'''test = Player()
test.SetResources({Resources.GOLD : ResourceStats(5,1), Resources.STONES : ResourceStats(3,2)})
hasResource,resourceStats = test.HasResource("gold")
test.DebugPrintResourceValueMap()'''