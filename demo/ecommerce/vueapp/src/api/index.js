import axios from 'axios';

export async function getAllProducts(user_id) {
  let result = await axios.post('/service/recommend/guess-you-like', {
    user_id,
    // user_id: 'A14EI4NEAWCH18',
  });
  return result.data.data;
}

export async function getOneProduct(user_id, item_id) {
  let result = await axios.post(`/service/itemSummary/item_id/${item_id}`, {
    user_id,
  });
  return result.data.data;
}

export async function LookAndLookProducts(user_id, item_id) {
  let result = await axios.post('/service/recommend/looked-and-looked', {
    user_id,
    item_id,
  });

  return result.data.data;
}
