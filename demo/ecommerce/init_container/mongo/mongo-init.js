print("Started Adding the Users.");
db = db.getSiblingDB("jpa");
db.createUser({
  user: "jpa",
  pwd: "test_mongodb_123456",
  roles: [{ role: "readWrite", db: "jpa" }],
});
print("End Adding the User Roles.");