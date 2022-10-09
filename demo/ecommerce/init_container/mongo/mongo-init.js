print("Started Adding the Users.");
db = db.getSiblingDB("jpa");
db.createUser({
  user: "jpa",
  pwd: "Dmetasoul_123456",
  roles: [{ role: "readWrite", db: "jpa" }],
});
print("End Adding the User Roles.");