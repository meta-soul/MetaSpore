File = io.open("requests.txt", "r");
Arr = {}
for line in File:lines() do
   table.insert (Arr, line);
end

request = function()
    local headers = {}
    headers["Content-Type"] = "application/json"
    local idx = math.floor(math.random(0, 9000)) + 1
    local body = Arr[idx]
    return wrk.format("POST", "/invocations", headers, body)
end

response = function(status, headers, body)
    if status ~= 200 then
      io.write("Status: ".. status .."\n")
      io.write("Body:\n")
      io.write(body .. "\n")
    end
end
